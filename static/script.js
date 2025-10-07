// FraudAlert: no client-side JS needed for page-based flow.
// This file is intentionally left blank.
const nameEl = document.getElementById('file_name');
if (nameEl){
  fileInput.addEventListener('change', () => {
    const f = fileInput.files && fileInput.files[0];
    nameEl.textContent = f ? f.name : '';
  });
}
