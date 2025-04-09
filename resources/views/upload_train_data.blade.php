<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Upload Train Data CSV</title>
  <!-- Import Google Font: Poppins -->
  <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;700&display=swap" rel="stylesheet">
  <style>
    /* Base Reset and Global Styles */
    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }
    body {
      font-family: 'Poppins', sans-serif;
      background: linear-gradient(135deg, #E0EAFC, #CFDEF3);
      min-height: 100vh;
      display: flex;
      justify-content: center;
      align-items: center;
      padding: 20px;
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
      transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .back-button-fixed:hover {
      transform: scale(1.03);
      box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    /* Modern Container Style */
    .container {
      background: #ffffff;
      padding: 30px 40px;
      border-radius: 12px;
      box-shadow: 0 8px 20px rgba(0,0,0,0.1);
      width: 100%;
      max-width: 500px;
      animation: fadeInUp 0.8s ease-out;
      margin-top: 70px; /* To ensure content isn't overlapped by the fixed back button */
    }
    @keyframes fadeInUp {
      from { opacity: 0; transform: translateY(20px); }
      to { opacity: 1; transform: translateY(0); }
    }
    h1 {
      text-align: center;
      margin-bottom: 1.5rem;
      font-weight: 700;
      color: #2F80ED;
    }
    form {
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }
    label {
      font-weight: 500;
      color: #333;
    }
    input[type="file"] {
      padding: 0.75rem;
      border: 1px solid #ccc;
      border-radius: 6px;
      font-size: 1rem;
      outline: none;
      transition: border-color 0.3s ease;
    }
    input[type="file"]:focus {
      border-color: #2D9CDB;
    }
    button {
      background: linear-gradient(135deg, #2D9CDB, #2F80ED);
      border: none;
      border-radius: 6px;
      color: #fff;
      padding: 0.75rem;
      font-size: 1rem;
      font-weight: 500;
      cursor: pointer;
      transition: transform 0.3s ease, box-shadow 0.3s ease;
      text-align: center;
    }
    button:hover {
      transform: scale(1.03);
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    /* Alert Messages */
    .alert {
      padding: 1rem;
      border-radius: 6px;
      margin-bottom: 1.5rem;
      text-align: center;
    }
    .alert-success {
      background: #d4edda;
      color: #155724;
    }
    .alert-error {
      background: #f8d7da;
      color: #721c24;
    }
  </style>
</head>
<body>
  <!-- Fixed Back Button -->
  <button class="back-button-fixed" onclick="window.location.href='{{ url('/') }}'">← Back to Upload</button>

  <div class="container">
    <h1>Upload Train Data CSV</h1>

    @if(session('success'))
      <div class="alert alert-success">{{ session('success') }}</div>
    @endif

    @if(session('error'))
      <div class="alert alert-error">{{ session('error') }}</div>
    @endif

    <form action="{{ route('train_data.upload') }}" method="POST" enctype="multipart/form-data">
      @csrf
      <label for="csv_file">Select CSV File:</label>
      <input type="file" id="csv_file" name="csv_file" required>
      <button type="submit">Upload</button>
    </form>
  </div>
</body>
</html>
