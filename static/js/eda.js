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

  upload2 = document.getElementById("upload-btn2");
  loading2 = document.getElementById("loading-btn2");
  loadingFunct2 = function () {
    upload2.classList.toggle("visible");
    upload2.classList.toggle("invisible");
      
    loading2.classList.toggle("visible");
    loading2.classList.toggle("invisible");
  };
  form2 = document.getElementById("form-variables");
  form2.addEventListener('submit', loadingFunct2);

  change = function () {
    upload2.classList.toggle("visible");
    upload2.classList.toggle("invisible");
  }
  
});