window.addEventListener("load", function(){
  upload = document.getElementById("upload-btn");
  loading = document.getElementById("loading-btn");

  loadingFunct = function () {
    upload.classList.toggle("visible");
    upload.classList.toggle("invisible");

    loading.classList.toggle("visible");
    loading.classList.toggle("invisible");
  };

  form = document.getElementById("form-file");
  form.addEventListener('submit', loadingFunct);
  
});