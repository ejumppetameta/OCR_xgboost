<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use App\Models\BankStatement;
use App\Models\Category;
use App\Models\TrainData;
use Illuminate\Support\Facades\Log;
use GuzzleHttp\Client;
use App\Http\Controllers\Traits\PdfOcrCommonTrait;

class PdfOcrController extends Controller
{
    use PdfOcrCommonTrait;

    // This property holds our categories records from the database.
    private $categories;

    public function process(Request $request)
    {
        // Load full category records (including 'keywords' and 'type')
        $this->categories = Category::all()->toArray();

        set_time_limit(0);
        $request->validate([
            'pdf' => 'required|mimes:pdf|max:20480',
        ]);

        // Store PDF in storage/app/pdfs
        $pdfPath = $request->file('pdf')->store('pdfs');
        $fullPdfPath = storage_path('app' . DIRECTORY_SEPARATOR . $pdfPath);

        if (!file_exists($fullPdfPath)) {
            return response()->json(['error' => "File not found"], 500);
        }

        // Convert PDF pages to images using Imagick
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

        // --- Parse Header Information ---
        $accountHolder = 'Unknown';
        $accountNumber = 'Unknown';
        $accountType = 'Unknown';
        $statementDate = null;

        if (preg_match('/\n?([A-Z ]+)\s+PENYATA AKAUN\s*\/\s*STATEMENT OF ACCOUNT/i', $extractedText, $match)) {
            $accountHolder = trim($match[1]);
        }
        if (preg_match('/Nombor Akaun\s*\/\s*Account Number\s*(\d+)/i', $extractedText, $match)) {
            $accountNumber = trim($match[1]);
        }
        if (preg_match('/Jenis Akaun\s*\/\s*Account Type\s*(.+)/i', $extractedText, $match)) {
            $accountType = trim($match[1]);
        }
        if (preg_match('/Tarikh Penyata\s*\/\s*Statement Date\s*(.+)/i', $extractedText, $match)) {
            $rawStatementDate = trim($match[1]);
            $statementDate = date('Y-m-d', strtotime($rawStatementDate));
        }
        $statementYear = date('Y', strtotime($statementDate));

        // Create Bank Statement record
        $statementRecord = BankStatement::create([
            'bank_name'       => 'Public Bank',
            'account_holder'  => $accountHolder,
            'account_number'  => $accountNumber,
            'account_type'    => $accountType,
            'closing_balance' => 0, // Will update after processing transactions
            'statement_date'  => $statementDate,
        ]);

        // --- Parse Transaction Table ---
        $lines = explode("\n", $extractedText);
        $transactionLines = [];
        $startCollecting = false;
        foreach ($lines as $line) {
            $trimmedLine = trim($line);
            if (stripos($trimmedLine, 'DATE') !== false &&
                stripos($trimmedLine, 'TRANSACTION') !== false &&
                stripos($trimmedLine, 'DEBIT') !== false &&
                stripos($trimmedLine, 'CREDIT') !== false &&
                stripos($trimmedLine, 'BALANCE') !== false
            ) {
                $startCollecting = true;
                continue;
            }
            if ($startCollecting) {
                if ($trimmedLine === '' || stripos($trimmedLine, 'Closing Balance') !== false) {
                    continue;
                }
                $transactionLines[] = $trimmedLine;
            }
        }
        Log::info('Transaction Lines Extracted: ' . json_encode($transactionLines));

        // --- Process Each Transaction Line Separately ---
        $lastTransactionDate = null;
        $lastTransaction = null; // This will hold the last saved Transaction record.
        foreach ($transactionLines as $line) {
            $line = trim($line);
            if ($line === '') {
                continue;
            }

            if (preg_match('/^(\d{1,2}\/\d{1,2}(?:\/\d{2,4})?)/', $line, $dateMatch)) {
                $dateToken = $dateMatch[1];
                if (!preg_match('/\/\d{2,4}$/', $dateToken)) {
                    $dateToken .= '/' . $statementYear;
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
            $description = substr(implode(' ', $descTokens), 0, 100);

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

                // Update lastTransaction to the latest saved Transaction record.
                $lastTransaction = $statementRecord->transactions()->latest()->first();
            } else {
                // If amounts are zero, merge description with the last transaction.
                if ($lastTransaction) {
                    $mergedDescription = trim($lastTransaction->description . ' ' . $description);
                    $lastTransaction->update([
                        'description' => substr($mergedDescription, 0, 100),
                        'category'    => $this->determineCategory($mergedDescription, $type, $this->categories)
                    ]);
                }
            }
        }

        // --- Update Statement's Closing Balance ---
        if (preg_match('/Closing Balance(?: In This Statement)?\s+([\d,]+\.\d{2})/i', $extractedText, $match)) {
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
            'message'   => 'Public Bank statement PDF processed successfully',
            'statement' => $statementRecord,
            'raw_text'  => $extractedText,
        ]);
    }
}
