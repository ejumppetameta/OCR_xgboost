<?php

namespace App\Http\Controllers;

use App\Models\BankStatement;
use Illuminate\Http\Request;
use PDF; // alias provided by barryvdh/laravel-dompdf
use Illuminate\Support\Facades\Response;

class StatementController extends Controller
{
    // Display the statement details view
    public function show($id)
    {
        // Eager load transactions related to the bank statement
        $statement = BankStatement::with('transactions')->findOrFail($id);
        return view('statement_details', compact('statement'));
    }

    // Generate and download a PDF of the bank statement details
    public function downloadPdf($id)
    {
        $statement = BankStatement::with('transactions')->findOrFail($id);

        // Load the PDF view. Create resources/views/pdf/statement_pdf.blade.php
        $pdf = PDF::loadView('pdf.statement_pdf', compact('statement'));

        return $pdf->download('statement_' . $id . '.pdf');
    }

    // Generate and download a CSV of the bank statement details
    public function downloadCsv($id)
    {
        $statement = BankStatement::with('transactions')->findOrFail($id);

        $headers = [
            'Content-Type'        => 'text/csv',
            'Content-Disposition' => 'attachment; filename=statement_' . $id . '.csv',
        ];

        $callback = function () use ($statement) {
            $handle = fopen('php://output', 'w');

            // Write header details (optional)
            fputcsv($handle, ['Account Holder', $statement->account_holder]);
            fputcsv($handle, ['Account Number', $statement->account_number]);
            fputcsv($handle, ['Account Type', $statement->account_type]);
            fputcsv($handle, ['Statement Date', $statement->statement_date]);
            fputcsv($handle, ['Closing Balance', number_format($statement->closing_balance, 2)]);
            fputcsv($handle, []); // empty row for separation

            // Write CSV header for transactions
            fputcsv($handle, ['Date', 'Description', 'Category', 'Debit', 'Credit', 'Balance', 'Type']);

            foreach ($statement->transactions as $transaction) {
                fputcsv($handle, [
                    $transaction->transaction_date,
                    $transaction->description,
                    $transaction->category,
                    number_format($transaction->debit, 2),
                    number_format($transaction->credit, 2),
                    number_format($transaction->balance, 2),
                    $transaction->type,
                ]);
            }

            fclose($handle);
        };

        return response()->stream($callback, 200, $headers);
    }
}
