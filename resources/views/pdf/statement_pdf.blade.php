<!-- resources/views/pdf/statement_pdf.blade.php -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Bank Statement PDF</title>
    <style>
        body { font-family: Arial, sans-serif; padding: 20px; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 10px; border: 1px solid #ddd; text-align: left; }
        th { background-color: #f4f4f4; }
        h1, h2 { text-align: center; }
    </style>
</head>
<body>
    <h1>Statement Details</h1>
    <div>
        <strong>Account Holder:</strong> {{ $statement->account_holder }}<br>
        <strong>Account Number:</strong> {{ $statement->account_number }}<br>
        <strong>Account Type:</strong> {{ $statement->account_type }}<br>
        <strong>Statement Date:</strong> {{ $statement->statement_date }}<br>
        <strong>Closing Balance:</strong> {{ number_format($statement->closing_balance, 2) }}
    </div>

    <h2>Transactions</h2>
    <table>
        <thead>
            <tr>
                <th>Date</th>
                <th>Description</th>
                <th>Category</th>
                <th>Debit</th>
                <th>Credit</th>
                <th>Balance</th>
                <th>Type</th>
            </tr>
        </thead>
        <tbody>
            @foreach($statement->transactions as $transaction)
                <tr>
                    <td>{{ $transaction->transaction_date }}</td>
                    <td>{{ $transaction->description }}</td>
                    <td>{{ $transaction->category }}</td>
                    <td>{{ number_format($transaction->debit, 2) }}</td>
                    <td>{{ number_format($transaction->credit, 2) }}</td>
                    <td>{{ number_format($transaction->balance, 2) }}</td>
                    <td>{{ $transaction->type }}</td>
                </tr>
            @endforeach
        </tbody>
    </table>
</body>
</html>
