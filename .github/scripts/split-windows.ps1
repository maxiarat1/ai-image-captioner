$inputFile = "ai-image-tagger-windows.zip"
$chunkSize = 1500MB
$stream = [System.IO.File]::OpenRead($inputFile)
$buffer = New-Object byte[] $chunkSize
$partNum = 0

while (($bytesRead = $stream.Read($buffer, 0, $buffer.Length)) -gt 0) {
    $outputFile = "ai-image-tagger-windows.zip.part-" + ([string]$partNum).PadLeft(2, '0')
    [System.IO.File]::WriteAllBytes($outputFile, $buffer[0..($bytesRead-1)])
    Write-Host "Created $outputFile"
    $partNum++
}

$stream.Close()
Write-Host "Split into $partNum parts"
