<?php
// Database connection parameters
$host = 'your_database_host';
$username = 'your_database_username';
$password = 'your_database_password';
$database = 'your_database_name';

// Create a database connection
$conn = new mysqli($host, $username, $password, $database);

// Check the connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

// Function to sanitize input data
function sanitize($data) {
    global $conn;
    return mysqli_real_escape_string($conn, trim($data));
}

// Process login form submission
if ($_SERVER['REQUEST_METHOD'] == 'POST' && isset($_POST['login-submit'])) {
    $email = sanitize($_POST['email']);
    $password = sanitize($_POST['password']);

    // Add your validation and authentication logic here

    // Example query (replace with your actual query)
    $query = "SELECT * FROM users WHERE email='$email' AND password='$password'";
    $result = $conn->query($query);

    if ($result->num_rows > 0) {
        // Login successful
        // Redirect or set session, etc.
    } else {
        // Login failed
        // Handle accordingly
    }
}

// Process signup form submission
if ($_SERVER['REQUEST_METHOD'] == 'POST' && isset($_POST['signup-submit'])) {
    $name = sanitize($_POST['name']);
    $email = sanitize($_POST['email']);
    $password = sanitize($_POST['password']);

    // Add your validation and registration logic here

    // Example query (replace with your actual query)
    $query = "INSERT INTO users (name, email, password) VALUES ('$name', '$email', '$password')";
    if ($conn->query($query) === TRUE) {
        // Registration successful
        // Redirect or set session, etc.
    } else {
        // Registration failed
        // Handle accordingly
    }
}
?>

<!-- Your HTML code goes here -->
<!DOCTYPE html>
<!-- ... (rest of your HTML code) ... -->
<body>
    <div class="forms">
        <div class="form-content">
            <div class="login-form">
                <div class="title">Login</div>
                <form method="post" action="<?php echo htmlspecialchars($_SERVER["PHP_SELF"]); ?>">
                    <!-- ... (your existing login form fields) ... -->
                    <div class="button input-box">
                        <input type="submit" value="Submit" name="login-submit">
                    </div>
                </form>
            </div>

            <div class="signup-form">
                <div class="title">Signup</div>
                <form method="post" action="<?php echo htmlspecialchars($_SERVER["PHP_SELF"]); ?>">
                    <!-- ... (your existing signup form fields) ... -->
                    <div class="button input-box">
                        <input type="submit" value="Submit" name="signup-submit">
                    </div>
                </form>
            </div>
        </div>
    </div>
</body>
</html>

<?php
// Close the database connection
$conn->close();
?>
