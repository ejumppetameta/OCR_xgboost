<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use App\Models\BankStatement;
use App\Models\Category;
use Illuminate\Support\Facades\Log;
use Exception;

class CsvVtableController extends Controller
{
    // This property holds our category records from the database.
    private $categories;

    use \App\Http\Controllers\Traits\PdfOcrCommonTrait;

    /**
     * Process a CSV bank statement and save all transactions only to the VTable.
     *
     * Expected CSV format:
     *   For the standard 5-column CSV:
     *     Header row: Date, Description, Debit, Credit, Balance
     *   For the extended CSV:
     *     Columns: Serial, ID, Date, Description, Category, Debit, Credit, Balance, etc.
     *
     * @param \Illuminate\Http\Request $request
     * @return \Illuminate\Http\JsonResponse
     */
    public function process(Request $request)
    {
        try {
            // Check that a CSV file is attached under the field 'csv'
            if (!$request->hasFile('csv')) {
                Log::error('CSV file missing from request');
                return response()->json(['error' => 'CSV file is required'], 400);
            }

            // Validate that a CSV file is uploaded.
            $request->validate([
                'csv' => 'required|mimes:csv,txt|max:20480',
            ]);

            // Load full category records.
            $this->categories = Category::all()->toArray();

            // Store CSV file in storage/app/csv.
            $csvPath = $request->file('csv')->store('csv');
            $fullCsvPath = storage_path('app' . DIRECTORY_SEPARATOR . $csvPath);

            if (!file_exists($fullCsvPath)) {
                Log::error('CSV file not found at path: ' . $fullCsvPath);
                return response()->json(['error' => "File not found"], 500);
            }

            if (($handle = fopen($fullCsvPath, 'r')) === false) {
                Log::error('Unable to open CSV file at path: ' . $fullCsvPath);
                return response()->json(['error' => 'Unable to open CSV file'], 500);
            }

            // Read the header row (assumes first row is header)
            $header = fgetcsv($handle);
            if (!$header || count($header) < 5) {
                Log::error('CSV header row is missing or does not have enough columns', ['header' => $header]);
                return response()->json(['error' => 'Invalid CSV header'], 500);
            }

            Log::info('CSV Header read:', $header);

            // Optionally, verify header names (case-insensitive).
            $expectedHeaders = ['date', 'description', 'debit', 'credit', 'balance'];
            $headerLower = array_map('strtolower', $header);
            foreach ($expectedHeaders as $expectedHeader) {
                if (!in_array($expectedHeader, $headerLower)) {
                    Log::warning("Expected header '{$expectedHeader}' not found in CSV", ['header' => $header]);
                }
            }

            // Create a BankStatement record.
            $statementRecord = BankStatement::create([
                'bank_name'       => 'CSV Bank',
                'account_holder'  => 'CSV Account Holder',
                'account_number'  => 'Unknown',
                'account_type'    => 'Unknown',
                'closing_balance' => 0, // Will update later.
                'statement_date'  => now()->format('Y-m-d'),
            ]);

            // Process each CSV row for transaction data.
            $rowsProcessed = 0;
            while (($data = fgetcsv($handle)) !== false) {
                // Skip empty rows.
                if (empty(array_filter($data))) {
                    continue;
                }

                Log::debug('CSV row data:', $data);

                // Check CSV structure – use correct column indices for each format.
                if (count($data) === 5) {
                    // Standard CSV format: [0] Date, [1] Description, [2] Debit, [3] Credit, [4] Balance.
                    $rawDate     = $data[0];
                    $description = trim($data[1]);
                    $debit       = floatval(str_replace([','], '', $data[2]));
                    $credit      = floatval(str_replace([','], '', $data[3]));
                    $balanceVal  = floatval(str_replace([','], '', $data[4]));
                } elseif (count($data) >= 8) {
                    // Extended CSV format: e.g., [0] Serial, [1] ID, [2] Date, [3] Description, [4] Category,
                    // [5] Debit, [6] Credit, [7] Balance, etc.
                    $rawDate     = $data[2];
                    $description = trim($data[3]);
                    $debit       = floatval(str_replace([','], '', $data[5]));
                    $credit      = floatval(str_replace([','], '', $data[6]));
                    $balanceVal  = floatval(str_replace([','], '', $data[7]));
                } else {
                    Log::warning('Row skipped because it does not have enough columns', $data);
                    continue;
                }

                // Convert the CSV date to Y-m-d format.
                // Try using a specific format first (e.g., 'd/m/Y'); adjust as necessary.
                $dateObj = \DateTime::createFromFormat('d/m/Y', $rawDate);
                if ($dateObj) {
                    $transactionDate = $dateObj->format('Y-m-d');
                } else {
                    // Fallback to strtotime in case the format is different.
                    $transactionDate = date('Y-m-d', strtotime($rawDate));
                }

                // Determine transaction type: Credit if any credit amount; otherwise Debit.
                $type = ($credit > 0) ? 'CR' : 'DR';

                // Use the shared determineCategory() logic from our trait.
                $finalCategory = $this->determineCategory($description, $type, $this->categories);
                $category = $finalCategory;

                // Optionally: skip transactions with an unreasonably high balance.
                if ($balanceVal > 1000000000) {
                    Log::info('Skipping transaction due to out-of-range balance: ' . $balanceVal);
                    continue;
                }

                Log::info('Parsed Transaction Data', [
                    'transaction_date' => $transactionDate,
                    'description'      => $description,
                    'category'         => $category,
                    'debit'            => $debit,
                    'credit'           => $credit,
                    'balance'          => $balanceVal,
                    'type'             => $type,
                ]);

                // Build the transaction data array.
                $transactionData = [
                    'transaction_date' => $transactionDate,
                    'description'      => $description,
                    'category'         => $category,
                    'debit'            => $debit,
                    'credit'           => $credit,
                    'balance'          => $balanceVal,
                    'type'             => $type,
                ];

                // Save the transaction only into the VTable.
                try {
                    $statementRecord->vtableRecords()->create($transactionData);
                    $rowsProcessed++;
                } catch (Exception $e) {
                    Log::error("Error saving transaction data to VTable: " . $e->getMessage(), [
                        'transaction_data' => $transactionData,
                    ]);
                }
            }

            fclose($handle);

            // Update the statement's closing balance based on the last VTable transaction.
            $lastVtableRecord = $statementRecord->vtableRecords()->orderBy('transaction_date', 'desc')->first();
            if ($lastVtableRecord) {
                $statementRecord->closing_balance = $lastVtableRecord->balance;
                $statementRecord->save();
            }

            Log::info("CSV processing completed. Total rows processed: {$rowsProcessed}");

            return response()->json([
                'message'   => 'CSV Bank statement processed successfully (data saved into VTable only)',
                'statement' => $statementRecord,
            ]);
        } catch (Exception $ex) {
            Log::error("Error processing CSV file: " . $ex->getMessage(), [
                'exception' => $ex,
            ]);
            return response()->json(['error' => 'Error processing CSV file'], 500);
        }
    }
}
