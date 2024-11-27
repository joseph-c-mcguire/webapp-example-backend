
# Ensure test_data.csv exists at the specified path before making the request
$testDataPath = "C:\Users\bigme\OneDrive\Documents\GitHub\webapp-example-backend\data\processed\test_data.csv"
if (-Not (Test-Path $testDataPath)) {
    Write-Error "Test data file not found at $testDataPath. Please ensure the file exists."
    exit
}

$uri = "http://127.0.0.1:5000/api/diagnostics/roc-curve"
$body = @{
    model_name = "Decision Tree"
    class_label = "No Failure"  # Ensure this is the correct positive class
} | ConvertTo-Json

Invoke-RestMethod -Uri $uri -Method POST -Body $body -ContentType "application/json"