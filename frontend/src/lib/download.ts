/** Hand a file to the browser's own download handling. */
export function saveFile(blob: Blob, fileName: string): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = fileName;
  link.hidden = true;
  document.body.append(link);
  link.click();
  link.remove();
  // The download has started from the URL; release it once the browser has taken it.
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
