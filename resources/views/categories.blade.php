<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Category Management</title>
  <!-- Import Bootstrap CSS for basic layout and modal support -->
  <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css">
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
    /* Fixed Back Button at top-left */
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
    .container {
      background: #ffffff;
      padding: 30px 40px;
      border-radius: 12px;
      box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
      max-width: 800px;
      width: 100%;
      animation: fadeInUp 0.8s ease-out;
      /* Add top margin to avoid overlap with fixed back button */
      margin-top: 70px;
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
    hr {
      border-top: 1px solid #ddd;
      margin-bottom: 1.5rem;
    }
    /* Card Styling */
    .card {
      border: none;
      border-radius: 12px;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
      margin-bottom: 1.5rem;
    }
    .card-header {
      background: none;
      border-bottom: 1px solid #eee;
      font-weight: 600;
      color: #2F80ED;
    }
    .card-body {
      padding: 1.5rem;
    }
    /* Form Elements */
    label {
      font-weight: 500;
      color: #333;
    }
    .form-control {
      border: 1px solid #ccc;
      border-radius: 6px;
      padding: 0.75rem;
      font-size: 1rem;
      transition: border-color 0.3s ease;
    }
    .form-control:focus {
      border-color: #2D9CDB;
      box-shadow: none;
    }
    /* Modern Button Styling */
    .btn {
      background: linear-gradient(135deg, #2D9CDB, #2F80ED);
      border: none;
      border-radius: 6px;
      color: #fff;
      font-weight: 500;
      transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .btn:hover {
      transform: scale(1.03);
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    /* Table Styling */
    table.table {
      border: none;
    }
    table.table th,
    table.table td {
      vertical-align: middle;
    }
    table.table thead th {
      background: #F3F9FF;
      color: #2F80ED;
      font-weight: 600;
      border: none;
    }
    table.table tbody tr {
      border-bottom: 1px solid #eee;
    }
    /* Modal Overrides */
    .modal-content {
      border-radius: 12px;
      box-shadow: 0 8px 20px rgba(0,0,0,0.1);
    }
    .modal-header {
      border-bottom: none;
      background: #F3F9FF;
    }
    .modal-title {
      color: #2F80ED;
      font-weight: 600;
    }
    .modal-footer button {
      min-width: 120px;
    }
    /* Margin top utility */
    .mt-20 {
      margin-top: 20px;
    }
  </style>
</head>
<body>
  <!-- Fixed Back Button -->
  <button class="back-button-fixed" onclick="window.location.href='{{ url('/') }}'">← Back to Upload</button>

  <div class="container">
    <h1>Category Management</h1>
    <hr>

    <!-- Add New Category Form -->
    <div class="card mb-4">
      <div class="card-header">Add New Category</div>
      <div class="card-body">
        <form id="add-category-form">
          <div class="form-group">
            <label for="name">Category Name:</label>
            <input type="text" name="name" id="name" class="form-control" required>
          </div>
          <div class="form-group">
            <label for="type">Type:</label>
            <input type="text" name="type" id="type" class="form-control" required>
          </div>
          <div class="form-group">
            <label for="keywords">Keywords (comma separated):</label>
            <input type="text" name="keywords" id="keywords" class="form-control" required>
          </div>
          <button type="submit" class="btn btn-primary">Add Category</button>
        </form>
      </div>
    </div>

    <!-- Category List -->
    <div class="card">
      <div class="card-header">Categories</div>
      <div class="card-body">
        <table class="table table-bordered" id="categories-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>Name</th>
              <th>Type</th>
              <th>Keywords</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            <!-- Categories will be loaded dynamically -->
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- Edit Category Modal -->
  <div class="modal fade" id="editCategoryModal" tabindex="-1" role="dialog" aria-labelledby="editCategoryModalLabel" aria-hidden="true">
    <div class="modal-dialog" role="document">
      <form id="edit-category-form">
        <div class="modal-content">
          <div class="modal-header">
            <h5 class="modal-title">Edit Category</h5>
            <button type="button" class="close" data-dismiss="modal" aria-label="Close">
              <span aria-hidden="true">&times;</span>
            </button>
          </div>
          <div class="modal-body">
            <input type="hidden" id="edit-category-id">
            <div class="form-group">
              <label for="edit-name">Category Name:</label>
              <input type="text" name="name" id="edit-name" class="form-control" required>
            </div>
            <div class="form-group">
              <label for="edit-type">Type:</label>
              <input type="text" name="type" id="edit-type" class="form-control" required>
            </div>
            <div class="form-group">
              <label for="edit-keywords">Keywords (comma separated):</label>
              <input type="text" name="keywords" id="edit-keywords" class="form-control" required>
            </div>
          </div>
          <div class="modal-footer">
            <button type="button" class="btn btn-secondary" data-dismiss="modal">Cancel</button>
            <button type="submit" class="btn btn-primary">Update Category</button>
          </div>
        </div>
      </form>
    </div>
  </div>

  <!-- Include required JS libraries -->
  <script src="https://code.jquery.com/jquery-3.5.1.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js"></script>
  <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js"></script>
  <script>
    // Function to load categories from the API
    function loadCategories() {
      axios.get('/api/categories')
        .then(function(response) {
          var categories = response.data;
          var tbody = $('#categories-table tbody');
          tbody.empty();
          if (Array.isArray(categories.data)) {
            categories.data.forEach(function(category) {
              var keywords = category.keywords.join(', ');
              var row = `
                <tr>
                  <td>${category.id}</td>
                  <td>${category.name}</td>
                  <td>${category.type}</td>
                  <td>${keywords}</td>
                  <td>
                    <button class="btn btn-sm btn-info edit-btn"
                      data-id="${category.id}"
                      data-name="${category.name}"
                      data-type="${category.type}"
                      data-keywords="${keywords}">
                      Edit
                    </button>
                    <button class="btn btn-sm btn-danger delete-btn" data-id="${category.id}">Delete</button>
                  </td>
                </tr>
              `;
              tbody.append(row);
            });
          } else {
            tbody.html('<tr><td colspan="5" class="text-center">No categories found.</td></tr>');
          }
        })
        .catch(function(error) {
          console.error(error);
          alert('Error loading categories.');
        });
    }

    $(document).ready(function() {
      // Load categories on page load
      loadCategories();

      // Handle add new category form submission
      $('#add-category-form').submit(function(e) {
        e.preventDefault();
        var name = $('#name').val();
        var type = $('#type').val();
        var keywords = $('#keywords').val().split(',').map(item => item.trim()).filter(item => item);
        axios.post('/api/categories', { name: name, type: type, keywords: keywords })
          .then(function(response) {
            alert(response.data.message);
            $('#add-category-form')[0].reset();
            loadCategories();
          })
          .catch(function(error) {
            console.error(error);
            alert('Error adding category.');
          });
      });

      // Handle click on edit button to open modal
      $(document).on('click', '.edit-btn', function() {
        var id = $(this).data('id');
        var name = $(this).data('name');
        var type = $(this).data('type');
        var keywords = $(this).data('keywords');
        $('#edit-category-id').val(id);
        $('#edit-name').val(name);
        $('#edit-type').val(type);
        $('#edit-keywords').val(keywords);
        $('#editCategoryModal').modal('show');
      });

      // Handle update category form submission
      $('#edit-category-form').submit(function(e) {
        e.preventDefault();
        var id = $('#edit-category-id').val();
        var name = $('#edit-name').val();
        var type = $('#edit-type').val();
        var keywords = $('#edit-keywords').val().split(',').map(item => item.trim()).filter(item => item);
        axios.put('/api/categories/' + id, { name: name, type: type, keywords: keywords })
          .then(function(response) {
            alert(response.data.message);
            $('#editCategoryModal').modal('hide');
            loadCategories();
          })
          .catch(function(error) {
            console.error(error);
            alert('Error updating category.');
          });
      });

      // Handle delete category button
      $(document).on('click', '.delete-btn', function() {
        if (!confirm('Are you sure you want to delete this category?')) {
          return;
        }
        var id = $(this).data('id');
        axios.delete('/api/categories/' + id)
          .then(function(response) {
            alert(response.data.message);
            loadCategories();
          })
          .catch(function(error) {
            console.error(error);
            alert('Error deleting category.');
          });
      });
    });
  </script>
</body>
</html>
