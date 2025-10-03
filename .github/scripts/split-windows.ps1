$inputFile = "ai-image-tagger-windows.zip"
$chunkSize = 1500MB
$inputStream = [System.IO.File]::OpenRead($inputFile)
$buffer = New-Object byte[] $chunkSize
$partNum = 0

while (($bytesRead = $inputStream.Read($buffer, 0, $buffer.Length)) -gt 0) {
    $outputFile = "ai-image-tagger-windows.zip.part-" + ([string]$partNum).PadLeft(2, '0')
    $outputStream = [System.IO.File]::OpenWrite($outputFile)
    $outputStream.Write($buffer, 0, $bytesRead)
    $outputStream.Close()
    Write-Host "Created $outputFile"
    $partNum++
}

$inputStream.Close()
Write-Host "Split into $partNum parts"
