<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use App\Models\BankStatement;
use App\Models\Category;
use Illuminate\Support\Facades\Log;
use GuzzleHttp\Client;
use App\Http\Controllers\Traits\PdfOcrCommonTrait;

class PdfOcrControllerCreditSense extends Controller
{
    use PdfOcrCommonTrait;

    private $categories;

    public function process(Request $request)
    {
        // Start output buffering to prevent stray output from corrupting JSON response.
        ob_start();

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
            ob_end_clean();
            return response()->json(['error' => "File not found"], 500);
        }

        // Create Imagick instance and set resource limits.
        try {
            $imagick = new \Imagick();
            $imagick->setResourceLimit(\Imagick::RESOURCETYPE_MEMORY, 1024); // 1024 MB
            $imagick->setResourceLimit(\Imagick::RESOURCETYPE_MAP, 1024);
            $imagick->setResolution(150, 150); // Lower resolution to reduce memory usage.
            $imagick->readImage($fullPdfPath);
        } catch (\Exception $e) {
            ob_end_clean();
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

        // --- Global Header Extraction ---
        $globalBankName = 'Credit Sense';
        if (preg_match('/Bank:\s*([^\s]+)/i', $extractedText, $matchBank)) {
            $headerBank = trim($matchBank[1]);
        } else {
            $headerBank = $globalBankName;
        }

        $statementDate = null;
        if (preg_match('/Report Period:\s*(\d{2}\/\d{2}\/\d{2})\s*-\s*(\d{2}\/\d{2}\/\d{2})/i', $extractedText, $matchPeriod)) {
            $statementDate = date('Y-m-d', strtotime($matchPeriod[2]));
        }

        $lines = explode("\n", $extractedText);

        // --- Reassemble Summary Table Rows ---
        $assembledRows = [];
        $currentRow = "";
        foreach ($lines as $line) {
            $trim = trim($line);
            if ($currentRow !== "" && substr($currentRow, -1) === "-") {
                $currentRow = rtrim($currentRow, "-") . $trim;
                continue;
            }
            if (preg_match('/^(?:\d+\s+)?\d+-/', $trim)) {
                if ($currentRow !== "") {
                    $assembledRows[] = $currentRow;
                }
                $currentRow = $trim;
            } else {
                if ($currentRow !== "") {
                    $currentRow .= " " . $trim;
                }
            }
        }
        if ($currentRow !== "") {
            $assembledRows[] = $currentRow;
        }
        Log::info('Assembled Summary Rows: ' . json_encode($assembledRows));

        // --- Parse Credit Summary Table ---
        $summaryAccounts = [];
        $regex = '/^(?:(?<index>\d+)\s+)?(?<account_number>(?:\d+-)+\d+)\s+(?<account_name>.+?)\s+(?<account_holder>[A-Za-z ]+)\s+\$\s+(?<available_balance>-?[\d,]+\.\d{2})\s+\$\s+(?<current_balance>-?[\d,]+\.\d{2})\s+\$\s+(?<total_debits>-?[\d,]+\.\d{2})\s+\$\s+(?<total_credits>-?[\d,]+\.\d{2})$/i';
        foreach ($assembledRows as $row) {
            if (preg_match($regex, $row, $match)) {
                $accountNumber = preg_replace('/\s+/', '', $match['account_number']);
                $summaryAccounts[] = [
                    'account_number'    => $accountNumber,
                    'account_type'      => trim($match['account_name']),
                    'account_holder'    => trim($match['account_holder']),
                    'available_balance' => floatval(str_replace([','], '', $match['available_balance'])),
                    'current_balance'   => floatval(str_replace([','], '', $match['current_balance'])),
                    'closing_balance'   => floatval(str_replace([','], '', $match['total_debits'])),
                    'total_credits'     => floatval(str_replace([','], '', $match['total_credits'])),
                ];
            }
        }
        Log::info('Extracted Summary Accounts: ' . json_encode($summaryAccounts));

        // --- Create BankStatement Records for Each Summary Account ---
        // Use normalized account numbers as keys
        $statementRecords = [];
        foreach ($summaryAccounts as $acc) {
            $record = BankStatement::create([
                'bank_name'       => $headerBank,
                'account_holder'  => $acc['account_holder'],
                'account_number'  => $acc['account_number'],
                'account_type'    => $acc['account_type'],
                'closing_balance' => $acc['closing_balance'],
                'statement_date'  => $statementDate,
            ]);
            $normalizedKey = $this->normalizeAccountNumber($acc['account_number']);
            $statementRecords[$normalizedKey] = $record;
        }

        // --- Determine the Account with Transaction Details ---
        $transactionAccountNumber = null;
        if (preg_match('/Statement:\s*(.*?)\s*-\s*\((.*?)\)/i', $extractedText, $matchStatement)) {
            $transactionAccountNumber = trim($matchStatement[2]);
        }
        $stmtRecord = null;
        if ($transactionAccountNumber) {
            $normalizedTrans = $this->normalizeAccountNumber($transactionAccountNumber);
            if (isset($statementRecords[$normalizedTrans])) {
                $stmtRecord = $statementRecords[$normalizedTrans];
            } else {
                foreach ($statementRecords as $acctNum => $record) {
                    // Use a substring match as fallback if necessary.
                    if (substr($acctNum, -strlen($normalizedTrans)) === $normalizedTrans) {
                        $stmtRecord = $record;
                        break;
                    }
                }
            }
        }
        // Fallback: if no matching transaction account identified, use the first summary account.
        if (!$stmtRecord && count($statementRecords) > 0) {
            $stmtRecord = reset($statementRecords);
            Log::warning("No matching transaction account found. Defaulting to first summary account.");
        }

        // --- Parse Transaction Table (Detailed) ---
        $transactionLines = [];
        $startCollecting = false;
        // Look for headers including Date, Description, Debit, Credit, Balance.
        foreach ($lines as $line) {
            $trimmedLine = trim($line);
            if (stripos($trimmedLine, 'Date') !== false &&
                stripos($trimmedLine, 'Description') !== false &&
                stripos($trimmedLine, 'Debit') !== false &&
                stripos($trimmedLine, 'Credit') !== false &&
                stripos($trimmedLine, 'Balance') !== false
            ) {
                $startCollecting = true;
                continue;
            }
            if ($startCollecting) {
                if ($trimmedLine === '' || stripos($trimmedLine, 'End Balance') !== false) {
                    continue;
                }
                $transactionLines[] = $trimmedLine;
            }
        }
        Log::info('Transaction Lines Extracted: ' . json_encode($transactionLines));

        // --- Process Each Transaction Line ---
        $lastTransactionDate = null;
        $stmtYear = date('Y', strtotime($statementDate));
        if ($stmtRecord) {
            foreach ($transactionLines as $line) {
                $line = trim($line);
                if ($line === '') {
                    continue;
                }
                // Use regex to capture a date token.
                $datePattern = '/\b(\d{2}\/\d{2}(?:\/\d{2,4})?)\b/';
                if (preg_match($datePattern, $line, $dateMatch)) {
                    $dateToken = $dateMatch[1];
                    if (substr_count($dateToken, '/') < 2) {
                        $dateToken .= '/' . $stmtYear;
                    }
                    if (preg_match('/^\d{2}\/\d{2}\/\d{2}$/', $dateToken)) {
                        $dateObj = \DateTime::createFromFormat('d/m/y', $dateToken);
                    } elseif (preg_match('/^\d{2}\/\d{2}\/\d{4}$/', $dateToken)) {
                        $dateObj = \DateTime::createFromFormat('d/m/Y', $dateToken);
                    } else {
                        $dateObj = false;
                    }
                    if ($dateObj) {
                        $parsedDate = $dateObj->format('Y-m-d');
                    } else {
                        $parsedDate = '1970-01-01';
                    }
                    if ($parsedDate === '1970-01-01' && $dateToken !== '01/01/1970') {
                        Log::warning("Parsed date resulted in Unix epoch for token '{$dateToken}', using previous date: " . ($lastTransactionDate ?? $statementDate));
                        $parsedDate = $lastTransactionDate ?? $statementDate;
                    }
                    $lastTransactionDate = $parsedDate;
                    $descriptionPart = trim(preg_replace($datePattern, '', $line, 1));
                } else {
                    if ($lastTransactionDate === null) {
                        Log::info('Skipping line with no date and no previous date: ' . $line);
                        continue;
                    }
                    $descriptionPart = $line;
                }

                // Tokenize to extract amounts and description.
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
                    $balanceVal = floatval(str_replace(',', '', $amounts[0]));
                } elseif (count($amounts) >= 3) {
                    $debit = floatval(str_replace(',', '', $amounts[0]));
                    $credit = floatval(str_replace(',', '', $amounts[1]));
                    $balanceVal = floatval(str_replace(',', '', end($amounts)));
                } elseif (count($amounts) == 2) {
                    if (stripos($descriptionPart, 'CR') !== false) {
                        $credit = floatval(str_replace(',', '', $amounts[0]));
                    } else {
                        $debit = floatval(str_replace(',', '', $amounts[0]));
                    }
                    $balanceVal = floatval(str_replace(',', '', $amounts[1]));
                }
                $description = substr(implode(' ', $descTokens), 0, 100);
                $type = ($credit > 0) ? 'CR' : 'DR';

                $finalCategory = $this->determineCategory($description, $type, $this->categories);
                $category = $finalCategory;

                if ($balanceVal > 1000000000) {
                    Log::info('Skipping transaction due to out-of-range balance: ' . $balanceVal);
                    continue;
                }

                Log::info('Parsed Transaction Data', [
                    'transaction_date' => $lastTransactionDate,
                    'description'      => $description,
                    'category'         => $category,
                    'debit'            => $debit,
                    'credit'           => $credit,
                    'balance'          => $balanceVal,
                    'type'             => $type,
                ]);

                $transactionData = [
                    'transaction_date' => $lastTransactionDate,
                    'description'      => $description,
                    'category'         => $category,
                    'debit'            => $debit,
                    'credit'           => $credit,
                    'balance'          => $balanceVal,
                    'type'             => $type,
                ];

                $uncertain = $this->autoDetermineUncertainFlag($transactionData, $this->categories);
                $this->saveTransactionData($stmtRecord, $transactionData, $uncertain);
            }

            // --- Update Statement's Closing Balance ---
            if (preg_match('/Closing Balance(?: In This Statement)?\s+([\d,]+\.\d{2})/i', $extractedText, $match)) {
                $closingBalance = floatval(str_replace(',', '', $match[1]));
                $stmtRecord->closing_balance = $closingBalance;
                $stmtRecord->save();
            } else {
                $lastTransactionRecord = $stmtRecord->transactions()->latest()->first();
                if ($lastTransactionRecord) {
                    $stmtRecord->closing_balance = $lastTransactionRecord->balance;
                    $stmtRecord->save();
                }
            }
        }

        ob_end_clean();
        return response()->json([
            'message'          => 'Credit Sense statement PDF processed successfully',
            'summary_accounts' => $summaryAccounts,
            'raw_text'         => $extractedText,
        ]);
    }

    /**
     * Normalizes an account number by removing all non-digit characters.
     *
     * @param string $acct
     * @return string
     */
    protected function normalizeAccountNumber($acct)
    {
        return preg_replace('/\D/', '', $acct);
    }
}
