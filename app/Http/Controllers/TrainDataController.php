<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Log;

class TrainDataController extends Controller
{
    /**
     * Show the CSV upload form.
     */
    public function index()
    {
        return view('upload_train_data');
    }

    /**
     * Process the CSV file and insert data into the train_data table.
     */
    public function upload(Request $request)
    {
        $request->validate([
            'csv_file' => 'required|file|mimes:csv,txt|max:20480', // max 20MB
        ]);

        $file = $request->file('csv_file');

        // Open the file for reading.
        if (($handle = fopen($file->getRealPath(), 'r')) !== false) {
            $header = null;
            $rowCount = 0;
            $inserted = 0;
            DB::beginTransaction();
            try {
                while (($row = fgetcsv($handle, 1000, ',')) !== false) {
                    // Assume first row is header.
                    if (!$header) {
                        $header = $row;
                        continue;
                    }

                    // Skip rows that do not match header count.
                    if (count($row) !== count($header)) {
                        Log::warning("Skipping row due to column mismatch: " . json_encode($row));
                        continue;
                    }

                    // Map row values to header columns.
                    $data = array_combine($header, $row);

                    // Validate that required columns exist.
                    if (!isset($data['description']) || !isset($data['category'])) {
                        Log::warning("Missing required columns in row: " . json_encode($data));
                        continue;
                    }

                    // If type column is missing or empty, generate one.
                    if (!isset($data['type']) || empty($data['type'])) {
                        $data['type'] = (stripos($data['description'], 'CR') !== false) ? 'CR' : 'DR';
                    }

                    // Insert into train_data table.
                    DB::table('train_data')->insert([
                        'description' => $data['description'],
                        'category'    => $data['category'],
                        'type'        => $data['type'],
                        'created_at'  => now(),
                        'updated_at'  => now(),
                    ]);
                    $inserted++;
                    $rowCount++;
                }
                DB::commit();
                fclose($handle);
                return redirect()->back()->with('success', "CSV processed successfully. Inserted {$inserted} rows.");
            } catch (\Exception $e) {
                DB::rollBack();
                fclose($handle);
                return redirect()->back()->with('error', "Error processing CSV: " . $e->getMessage());
            }
        } else {
            return redirect()->back()->with('error', 'Unable to open the file.');
        }
    }
}
