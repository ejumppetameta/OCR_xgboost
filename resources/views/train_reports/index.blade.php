<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Training Evaluation Reports</title>
  <!-- Import Google Font: Poppins -->
  <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;700&display=swap" rel="stylesheet">
  <style>
    /* Base Reset and styling similar to your upload page */
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
    table {
      width: 100%;
      border-collapse: collapse;
      margin-bottom: 1.5rem;
    }
    th, td {
      padding: 10px;
      border-bottom: 1px solid #ccc;
      text-align: left;
    }
    th {
      background: #f4f4f4;
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
    .pagination {
      display: flex;
      list-style: none;
      justify-content: center;
      padding: 0;
    }
    .pagination li {
      margin: 0 5px;
    }
    .pagination li a {
      padding: 5px 10px;
      border: 1px solid #ccc;
      border-radius: 4px;
      color: #2F80ED;
      text-decoration: none;
    }
    .pagination li.active span {
      background: #2F80ED;
      color: #fff;
      padding: 5px 10px;
      border-radius: 4px;
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
  <button class="back-button-fixed" onclick="window.location.href='{{ url('/') }}'">← Back to Home</button>

  <div class="container">
    <h1>Training Evaluation Reports</h1>
    @if($reports->count())
      <table>
        <thead>
          <tr>
            <th>ID</th>
            <th>Evaluation Type</th>
            <th>Created At</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          @foreach($reports as $report)
            <tr>
              <td>{{ $report->id }}</td>
              <td>{{ $report->evaluation_type }}</td>
              <td>{{ $report->created_at }}</td>
              <td>
                <a href="{{ route('train_reports.show', $report->id) }}" class="btn">View</a>
              </td>
            </tr>
          @endforeach
        </tbody>
      </table>
      {{ $reports->links() }}
    @else
      <p>No evaluation reports found.</p>
    @endif
  </div>
</body>
</html>
