<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use App\Models\BankStatement;
use App\Models\Category;
use Illuminate\Support\Facades\Log;
use GuzzleHttp\Client;
use App\Http\Controllers\Traits\PdfOcrCommonTrait;

class PdfOcrControllerMaybank extends Controller
{
    use PdfOcrCommonTrait;

    private $categories;

    public function process(Request $request)
    {
        // Load full category records (including 'keywords' and 'type')
        $this->categories = Category::all()->toArray();

        set_time_limit(0);
        $request->validate([
            'pdf' => 'required|mimes:pdf|max:20480',
        ]);

        // Store PDF in storage/app/pdfs.
        $pdfPath = $request->file('pdf')->store('pdfs');
        $fullPdfPath = storage_path('app' . DIRECTORY_SEPARATOR . $pdfPath);

        if (!file_exists($fullPdfPath)) {
            return response()->json(['error' => "File not found"], 500);
        }

        // Convert PDF pages to images using Imagick.
        try {
            $imagick = new \Imagick();
            $imagick->setResolution(300, 300);
            $imagick->readImage($fullPdfPath);
        } catch (\Exception $e) {
            return response()->json(['error' => 'Error reading PDF: ' . $e->getMessage()], 500);
        }

        // Create temporary images directory if needed.
        $tempDir = storage_path('app' . DIRECTORY_SEPARATOR . 'temp_images');
        if (!is_dir($tempDir)) {
            mkdir($tempDir, 0777, true);
        }

        $extractedText = "";
        foreach ($imagick as $i => $page) {
            try {
                // Pre-process image: grayscale, contrast, normalize, set format to png.
                $page->setImageColorspace(\Imagick::COLORSPACE_GRAY);
                $page->contrastImage(1);
                $page->normalizeImage();
                $page->setImageFormat('png');

                $tempImagePath = $tempDir . "/temp_page_{$i}.png";
                $page->writeImage($tempImagePath);

                $tesseractCommand = "tesseract " . escapeshellarg($tempImagePath) . " stdout --psm 6 -l eng";
                $pageText = shell_exec($tesseractCommand);
                $extractedText .= $pageText . "\n";
            } catch (\Exception $e) {
                Log::error("OCR error on page {$i}: " . $e->getMessage());
            } finally {
                if (isset($tempImagePath) && file_exists($tempImagePath)) {
                    unlink($tempImagePath);
                }
            }
        }
        $imagick->destroy();

        // --- Parse Header Information (Maybank version) ---
        $accountHolder = 'Unknown';
        $accountNumber = 'Unknown';
        $accountType = 'Savings';
        $statementDate = null;

        if (preg_match('/TARIKH PENYATA\s*\n([A-Z\s]+)(?:\s|$)/m', $extractedText, $match)) {
            $accountHolder = trim($match[1]);
        }
        if (preg_match('/TARIKH PENYATA\s*\n[A-Z\s]+.*?(\d{2}\/\d{2}\/\d{2})/m', $extractedText, $match)) {
            $rawDate = trim($match[1]);
            $parts = explode('/', $rawDate);
            if (count($parts) == 3) {
                list($day, $month, $year) = $parts;
                if (strlen($year) == 2) {
                    $year = '20' . $year;
                }
                $statementDate = "$year-$month-$day";
            } else {
                $statementDate = date('Y-m-d', strtotime($rawDate));
            }
        }
        if (preg_match('/(\d{6,}-\d{6,})\s*\nNUMBER/i', $extractedText, $match)) {
            $accountNumber = trim($match[1]);
        }
        if (preg_match('/PROTECTED BY PIDM UP TO RM250,000 FOR EACH DEPOSITOR\s+([A-Z0-9\s\-]+)/i', $extractedText, $match)) {
            $rawAccountType = trim($match[1]);
            $accountTypeParts = explode("\n", $rawAccountType);
            $accountType = trim($accountTypeParts[0]);
        }

        // Create Bank Statement record.
        $statementRecord = BankStatement::create([
            'bank_name'       => 'Maybank Berhad',
            'account_holder'  => $accountHolder,
            'account_number'  => $accountNumber,
            'account_type'    => $accountType,
            'closing_balance' => 0, // Will update after processing transactions.
            'statement_date'  => $statementDate,
        ]);

        // --- Parse Transaction Table ---
        $lines = explode("\n", $extractedText);
        $transactionLines = [];
        $startCollecting = false;
        foreach ($lines as $line) {
            $trimmedLine = trim($line);
            if (
                stripos($trimmedLine, 'ENTRY DATE') !== false &&
                stripos($trimmedLine, 'TRANSACTION DESCRIPTION') !== false &&
                stripos($trimmedLine, 'TRANSACTION AMOUNT') !== false &&
                stripos($trimmedLine, 'STATEMENT BALANCE') !== false
            ) {
                $startCollecting = true;
                continue;
            }
            if ($startCollecting) {
                if (
                    stripos($trimmedLine, 'ENDING BALANCE') !== false ||
                    stripos($trimmedLine, 'TOTAL CREDIT') !== false ||
                    stripos($trimmedLine, 'TOTAL DEBIT') !== false ||
                    stripos($trimmedLine, 'BEGINNING BALANCE') !== false ||
                    $trimmedLine === ''
                ) {
                    continue;
                }
                $transactionLines[] = $trimmedLine;
            }
        }
        Log::info('Transaction Lines Extracted: ' . json_encode($transactionLines));

        // --- Process Each Transaction ---
        $lastTransactionDate = null;
        $lastTransaction = null; // Holds the last saved Transaction record.
        foreach ($transactionLines as $line) {
            $line = trim($line);
            if ($line === '') {
                continue;
            }

            if (preg_match('/^(\d{1,2}\/\d{1,2}(?:\/\d{2,4})?)/', $line, $dateMatch)) {
                $dateToken = $dateMatch[1];
                if (!preg_match('/\/\d{2,4}$/', $dateToken)) {
                    $dateToken .= '/' . date('Y', strtotime($statementDate));
                }
                $lastTransactionDate = date('Y-m-d', strtotime($dateToken));
                $descriptionPart = trim(substr($line, strlen($dateMatch[1])));
            } else {
                if ($lastTransactionDate === null) {
                    Log::info('Skipping line with no date and no previous date: ' . $line);
                    continue;
                }
                $descriptionPart = $line;
            }

            $tokens = explode(' ', $descriptionPart);
            $amounts = [];
            $descTokens = [];
            $amountRegex = '/^(\d{1,3}(?:,\d{3})*\.\d{2})([A-Za-z].*)?$/';

            foreach ($tokens as $token) {
                if (preg_match($amountRegex, $token, $matchAmt)) {
                    $amounts[] = $matchAmt[1];
                    if (isset($matchAmt[2]) && trim($matchAmt[2]) !== '') {
                        $descTokens[] = trim($matchAmt[2]);
                    }
                } else {
                    $descTokens[] = $token;
                }
            }

            $debit = 0.0;
            $credit = 0.0;
            $balanceVal = 0.0;
            if (count($amounts) == 1) {
                $balanceVal = floatval(str_replace([','], '', $amounts[0]));
            } elseif (count($amounts) >= 3) {
                $debit = floatval(str_replace([','], '', $amounts[0]));
                $credit = floatval(str_replace([','], '', $amounts[1]));
                $balanceVal = floatval(str_replace([','], '', end($amounts)));
            } elseif (count($amounts) == 2) {
                if (stripos($descriptionPart, 'CR') !== false) {
                    $credit = floatval(str_replace([','], '', $amounts[0]));
                } else {
                    $debit = floatval(str_replace([','], '', $amounts[0]));
                }
                $balanceVal = floatval(str_replace([','], '', $amounts[1]));
            }
            $description = substr(implode(' ', $descTokens), 0, 255);

            // --- Determine transaction type ---
            $type = ($credit > 0) ? 'CR' : 'DR';

            // --- Determine the Category Using the Shared Functions ---
            $finalCategory = $this->determineCategory($description, $type, $this->categories);
            $category = $finalCategory;

            if ($balanceVal > 1000000000) {
                Log::info('Skipping transaction due to out-of-range balance: ' . $balanceVal);
                continue;
            }

            Log::info('Parsed Transaction Data', [
                'date'        => $lastTransactionDate,
                'description' => $description,
                'category'    => $category,
                'debit'       => $debit,
                'credit'      => $credit,
                'balance'     => $balanceVal,
                'type'        => $type,
            ]);

            // If amounts are non-zero, process as a new transaction.
            if ($debit != 0.0 || $credit != 0.0 || $balanceVal != 0.0) {
                // Build the transaction data array.
                $transactionData = [
                    'transaction_date' => $lastTransactionDate,
                    'description'      => $description,
                    'category'         => $category,
                    'debit'            => $debit,
                    'credit'           => $credit,
                    'balance'          => $balanceVal,
                    'type'             => $type,
                ];

                // Auto-determine the uncertain flag using the transaction data.
                $uncertain = $this->autoDetermineUncertainFlag($transactionData, $this->categories);

                // Save the transaction data accordingly.
                $this->saveTransactionData($statementRecord, $transactionData, $uncertain);

                // Update $lastTransaction to the latest saved Transaction record.
                $lastTransaction = $statementRecord->transactions()->latest()->first();
            } else {
                // If amounts are zero, merge description with the last transaction.
                if ($lastTransaction) {
                    $mergedDescription = trim($lastTransaction->description . ' ' . $description);
                    $lastTransaction->update([
                        'description' => substr($mergedDescription, 0, 255),
                        'category'    => $this->determineCategory($mergedDescription, $type, $this->categories)
                    ]);
                }
            }
        }

        // --- Update Statement's Closing Balance ---
        // Look for closing balance from "ENDING BALANCE :"
        if (preg_match('/ENDING BALANCE\s*:\s*([\d,]+\.\d{2})/i', $extractedText, $match)) {
            $closingBalance = floatval(str_replace([','], '', $match[1]));
            $statementRecord->closing_balance = $closingBalance;
            $statementRecord->save();
        } else {
            $lastTransactionRecord = $statementRecord->vtableRecords()->orderBy('transaction_date', 'desc')->first();
            if ($lastTransactionRecord) {
                $statementRecord->closing_balance = $lastTransactionRecord->balance;
                $statementRecord->save();
            }
        }

        return response()->json([
            'message'   => 'Maybank statement PDF processed successfully',
            'statement' => $statementRecord,
            'raw_text'  => $extractedText,
        ]);
    }
}
