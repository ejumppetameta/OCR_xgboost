<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Report Details</title>
  <!-- Import Google Font: Poppins -->
  <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;700&display=swap" rel="stylesheet">
  <style>
    /* Base Reset and styling similar to the list view */
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Poppins', sans-serif;
      background: linear-gradient(135deg, #E0EAFC, #CFDEF3);
      min-height: 100vh;
      display: flex;
      justify-content: center;
      align-items: center;
      padding: 20px;
    }
    .container {
      background: #fff;
      padding: 30px 40px;
      border-radius: 12px;
      box-shadow: 0 8px 20px rgba(0,0,0,0.1);
      width: 100%;
      max-width: 600px;
      margin-top: 70px;
    }
    h1 {
      text-align: center;
      margin-bottom: 1.5rem;
      font-weight: 700;
      color: #2F80ED;
    }
    h3, h4, p {
      margin-bottom: 0.75rem;
    }
    pre {
      background: #f4f4f4;
      padding: 15px;
      border-radius: 6px;
      overflow-x: auto;
      white-space: pre-wrap;
    }
    .btn {
      background: linear-gradient(135deg, #2D9CDB, #2F80ED);
      border: none;
      border-radius: 6px;
      color: #fff;
      padding: 0.5rem 0.75rem;
      cursor: pointer;
      text-decoration: none;
      font-size: 0.9rem;
    }
    .btn:hover {
      transform: scale(1.03);
      box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    /* Fixed Back Button */
    .back-button-fixed {
      position: fixed;
      top: 20px;
      left: 20px;
      background: #6c757d;
      color: #fff;
      border: none;
      border-radius: 6px;
      padding: 5px 10px;
      font-size: 0.85rem;
      cursor: pointer;
      z-index: 1100;
      transition: transform 0.3s ease;
    }
    .back-button-fixed:hover {
      transform: scale(1.03);
      box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
  </style>
</head>
<body>
  <!-- Fixed Back Button -->
  <button class="back-button-fixed" onclick="window.location.href='{{ route('train_reports.index') }}'">← Back to Reports</button>

  <div class="container">
    <h1>Report Details</h1>
    <h3>ID: {{ $report->id }}</h3>
    <h4>Type: {{ $report->evaluation_type }}</h4>
    <p><strong>Created At:</strong> {{ $report->created_at }}</p>
    <pre>{{ $report->report }}</pre>
    <a href="{{ route('train_reports.index') }}" class="btn">Back to Reports</a>
  </div>
</body>
</html>
