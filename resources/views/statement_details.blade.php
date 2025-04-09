<!-- resources/views/statement_details.blade.php -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Bank Statement Details</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/normalize/8.0.1/normalize.css">
    <style>
        body { font-family: Arial, sans-serif; padding: 20px; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 10px; border: 1px solid #ddd; text-align: left; }
        th { background-color: #f4f4f4; }
        .container { max-width: 1000px; margin: auto; }
        h1 { text-align: center; }
        /* Download button styles */
        .btn-group {
            text-align: center;
            margin: 20px 0;
        }
        .btn {
            display: inline-block;
            padding: 10px 20px;
            margin: 0 10px;
            background-color: #2F80ED;
            color: #fff;
            text-decoration: none;
            border-radius: 4px;
            transition: background-color 0.3s ease;
        }
        .btn:hover {
            background-color: #1c5bbf;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Statement Details</h1>
        <div>
            <strong>Account Holder:</strong> {{ $statement->account_holder }}<br>
            <strong>Account Number:</strong> {{ $statement->account_number }}<br>
            <strong>Account Type:</strong> {{ $statement->account_type }}<br>
            <strong>Statement Date:</strong> {{ $statement->statement_date }}<br>
            <strong>Closing Balance:</strong> {{ number_format($statement->closing_balance, 2) }}
        </div>

        <!-- Download buttons -->
        <div class="btn-group">
            <a href="{{ route('download.pdf', $statement->id) }}" class="btn">Download as PDF</a>
            <a href="{{ route('download.csv', $statement->id) }}" class="btn">Download as CSV</a>
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
    </div>
</body>
</html>
