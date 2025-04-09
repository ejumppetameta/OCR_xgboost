<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <!-- Ensure proper scaling on mobile devices -->
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Financial Dashboard - Bank Statement OCR</title>
  <!-- Import Google Font: Poppins -->
  <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;700&display=swap" rel="stylesheet">
  <style>
    /* Base Reset and global styles */
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
    /* Container style for main content */
    .container {
      background: #ffffff;
      padding: 30px 40px;
      border-radius: 12px;
      box-shadow: 0 8px 20px rgba(0,0,0,0.1);
      width: 100%;
      max-width: 500px;
      animation: fadeInUp 0.8s ease-out;
      position: relative;
      z-index: 1;
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
    select, input[type="file"] {
      padding: 0.75rem;
      border: 1px solid #ccc;
      border-radius: 6px;
      font-size: 1rem;
      outline: none;
      transition: border-color 0.3s ease;
    }
    select:focus, input[type="file"]:focus {
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
    /* Progress and result styles */
    #pleaseWaitContainer {
      text-align: center;
      margin-top: 10px;
      display: none;
    }
    #pleaseWaitText, #progressPercent {
      font-size: 1.2rem;
      color: #2F80ED;
      animation: blinkFade 1.5s infinite;
    }
    @keyframes blinkFade {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.5; }
    }
    .progress-container {
      width: 100%;
      height: 10px;
      background: #eee;
      border-radius: 6px;
      overflow: hidden;
      margin-top: 10px;
    }
    .progress-bar {
      height: 100%;
      width: 0%;
      background: linear-gradient(135deg, #2D9CDB, #2F80ED);
      transition: width 0.2s ease;
    }
    #result {
      margin-top: 1.5rem;
      padding: 1rem;
      border: 1px solid #e1e8ed;
      background: #f9f9f9;
      border-radius: 6px;
      white-space: pre-wrap;
      display: none;
      max-height: 300px;
      overflow-y: auto;
      font-size: 0.9rem;
      color: #333;
    }
    .btn-group {
      text-align: center;
      margin-top: 1.5rem;
    }
    /* Hidden Sidebar for Navigation */
    .sidebar {
      position: fixed;
      top: 0;
      left: -300px;
      width: 300px;
      height: 100%;
      background: #fff;
      box-shadow: 2px 0 12px rgba(0,0,0,0.2);
      transition: left 0.3s ease;
      padding: 80px 20px 20px 20px;
      z-index: 1000;
    }
    .sidebar.active {
      left: 0;
    }
    .sidebar h2 {
      margin-bottom: 1rem;
      color: #2F80ED;
      text-align: center;
    }
    .sidebar ul {
      list-style: none;
      padding-left: 0;
    }
    .sidebar li {
      padding: 8px 0;
      border-bottom: 1px solid #eee;
      cursor: pointer;
      text-align: center;
    }
    .sidebar li:hover {
      background: #f1f1f1;
    }
    /* Buttons inside sidebar */
    .sidebar .categories-button,
    .sidebar .extra-button {
      display: block;
      width: 100%;
      margin-top: 20px;
      background: linear-gradient(135deg, #2D9CDB, #2F80ED);
      border: none;
      border-radius: 6px;
      color: #fff;
      padding: 10px;
      text-align: center;
      cursor: pointer;
      transition: background 0.3s ease;
    }
    .sidebar .categories-button:hover,
    .sidebar .extra-button:hover {
      background: #2563A8;
    }
    /* Toggle button for sidebar */
    .toggle-sidebar {
      position: fixed;
      top: 20px;
      left: 20px;
      background: #2F80ED;
      color: #fff;
      padding: 10px 15px;
      border: none;
      border-radius: 6px;
      cursor: pointer;
      z-index: 1100;
      transition: background 0.3s ease;
    }
    .toggle-sidebar:hover {
      background: #2563A8;
    }
    /* Optional: move toggle button when sidebar is open */
    .sidebar.active + .toggle-sidebar {
      left: 320px;
    }
    @media (max-width: 600px) {
      .container {
        padding: 20px;
      }
    }
  </style>
</head>
<body>
  <!-- Sidebar Toggle Button -->
  <button class="toggle-sidebar" onclick="toggleSidebar()">☰ Menu</button>

  <!-- Sidebar -->
  <div class="sidebar" id="sidebar">
    <h2>Manage Categories</h2>
    <!-- "Go to Categories Page" button loads the categories.blade.php view -->
    <button class="categories-button" onclick="window.location.href='{{ url('/categories') }}'">Go to Categories Page</button>
    <!-- "Train Data" button with original URL -->
    <button class="extra-button" onclick="window.location.href='/train-data/upload'">Train Data</button>
  </div>

  <!-- Main Container -->
  <div class="container">
    <h1>Upload Bank Statement</h1>
    <form id="uploadForm" action="" method="POST" enctype="multipart/form-data">
      @csrf
      <label for="pdfType">Select Statement Type:</label>
      <select id="pdfType" name="pdfType">
        <option value="public">Public Bank Statement (PDF)</option>
        <option value="creditSense">Credit Sense Statement (PDF)</option>
        <option value="maybank">Maybank Statement (PDF)</option>
        <option value="csv">CSV Statement</option>
      </select>

      <!-- The file input will dynamically update its name and label -->
      <label id="fileLabel" for="fileInput">Select PDF:</label>
      <input type="file" id="fileInput" name="pdf" required>

      <button type="submit">Upload and Process</button>

      <!-- Progress bar container -->
      <div id="pleaseWaitContainer">
        <div id="pleaseWaitText">
          Please wait... AI in progress <span id="progressPercent">0%</span>
        </div>
        <div class="progress-container">
          <div class="progress-bar" id="progressBar"></div>
        </div>
      </div>
    </form>

    <!-- View Statement button (initially hidden) -->
    <div id="viewStatementDiv" class="btn-group" style="display: none;">
      <button id="viewStatementBtn" type="button">
        View Statement Details
      </button>
    </div>

    <!-- Result container -->
    <div id="result"></div>
  </div>

  <script>
    // Sidebar toggle function
    function toggleSidebar() {
      const sidebar = document.getElementById('sidebar');
      sidebar.classList.toggle('active');
    }
    // Auto-hide sidebar when mouse leaves its area
    document.getElementById('sidebar').addEventListener('mouseleave', function() {
      this.classList.remove('active');
    });

    // Update form action and file input attributes based on statement type selection
    const form = document.getElementById('uploadForm');
    const pdfTypeSelect = document.getElementById('pdfType');
    const fileInput = document.getElementById('fileInput');
    const fileLabel = document.getElementById('fileLabel');

    function updateFormAction() {
      const type = pdfTypeSelect.value;
      if (type === 'creditSense') {
        form.action = "{{ route('process-pdf-credit-sense') }}";
      } else if (type === 'maybank') {
        form.action = "{{ route('process-pdf-maybank') }}";
      } else if (type === 'csv') {
        // CSV processing route; update file input name and label.
        form.action = "{{ route('process-csv') }}";
        fileInput.name = "csv";
        fileLabel.textContent = "Select CSV File:";
        // Optionally adjust accepted file types:
        fileInput.accept = ".csv, .txt";
      } else {
        form.action = "{{ route('process-pdf-public') }}";
      }
      // For PDF types, ensure the input name is 'pdf' and accepted type is pdf
      if (type !== 'csv') {
        fileInput.name = "pdf";
        fileLabel.textContent = "Select PDF:";
        fileInput.accept = "application/pdf";
      }
    }
    // Initialize form action and file input attributes on page load
    updateFormAction();
    pdfTypeSelect.addEventListener('change', updateFormAction);

    // Helper function to update the progress bar.
    function updateProgressBar(percent) {
      const progressBar = document.getElementById('progressBar');
      const progressPercent = document.getElementById('progressPercent');
      progressBar.style.width = percent + '%';
      progressPercent.textContent = percent + '%';
    }

    // AJAX form submission for file upload
    form.addEventListener('submit', function(e) {
      e.preventDefault();
      const pleaseWaitContainer = document.getElementById('pleaseWaitContainer');
      pleaseWaitContainer.style.display = 'block';
      updateProgressBar(0);

      const formData = new FormData(form);
      const xhr = new XMLHttpRequest();
      xhr.open('POST', form.action, true);
      xhr.setRequestHeader('X-CSRF-TOKEN', '{{ csrf_token() }}');

      // Monitor upload progress (first 50%)
      xhr.upload.onprogress = function(e) {
        if (e.lengthComputable) {
          let percentComplete = Math.round((e.loaded / e.total) * 50);
          updateProgressBar(percentComplete);
        }
      };

      xhr.onreadystatechange = function() {
        if (xhr.readyState === XMLHttpRequest.DONE) {
          let data;
          try {
            data = JSON.parse(xhr.responseText);
          } catch(err) {
            data = { error: 'Invalid JSON response' };
          }
          let progress = 50;
          let interval = setInterval(function() {
            if (progress < 100) {
              progress += 2;
              updateProgressBar(progress);
            } else {
              clearInterval(interval);
            }
          }, 100);
          setTimeout(function() {
            pleaseWaitContainer.style.display = 'none';
          }, 2000);
          document.getElementById('result').style.display = 'block';
          document.getElementById('result').innerText = JSON.stringify(data, null, 2);

          if (data.statement && data.statement.id && data.statement.closing_balance > 0) {
            const statementId = data.statement.id;
            document.getElementById('viewStatementBtn').onclick = function() {
              window.open(`{{ url('/statement-details') }}/${statementId}`, '_blank');
            };
            document.getElementById('viewStatementDiv').style.display = 'block';
          }
        }
      };

      xhr.onerror = function() {
        pleaseWaitContainer.style.display = 'none';
        document.getElementById('result').style.display = 'block';
        document.getElementById('result').innerText = 'Error: ' + xhr.statusText;
      };

      xhr.send(formData);
    });
  </script>
</body>
</html>
