param(
    [Parameter(Mandatory=$true)][string]$Node,
    [string]$Group = '🔰 选择节点'
)

$pipeName = 'verge-mihomo-sidecar-release-a6e33b557bff1c853502bf0943e4d0bde463479a4d3d93730151672311b06e85'
$pipe = New-Object System.IO.Pipes.NamedPipeClientStream('.', $pipeName, [System.IO.Pipes.PipeDirection]::InOut)
try {
    $pipe.Connect(3000)
    $body = '{"name":"' + $Node.Replace('\\', '\\\\').Replace('"', '\\"') + '"}'
    $bodyBytes = [Text.Encoding]::UTF8.GetBytes($body)
    $encodedGroup = [Uri]::EscapeDataString($Group)
    $request = "PUT /proxies/$encodedGroup HTTP/1.1`r`nHost: localhost`r`nConnection: close`r`nContent-Type: application/json`r`nContent-Length: $($bodyBytes.Length)`r`n`r`n"
    $requestBytes = [Text.Encoding]::ASCII.GetBytes($request)
    $pipe.Write($requestBytes, 0, $requestBytes.Length)
    $pipe.Write($bodyBytes, 0, $bodyBytes.Length)
    $pipe.Flush()
    $reader = New-Object IO.StreamReader($pipe, [Text.Encoding]::UTF8)
    $response = $reader.ReadToEnd()
    if ($response -notmatch '204 No Content') { throw "Clash API did not accept node selection: $response" }
} finally {
    $pipe.Dispose()
}
