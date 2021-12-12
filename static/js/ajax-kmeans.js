$(document).ready(function () {
  function upload(event) {
    event.preventDefault();
    var data = new FormData($('#form-file').get(0));
    
    $.ajax({
      url: $(this).attr('action'),
      type: $(this).attr('method'),
      data: data,
      cache: false,
      processData: false,
      contentType: false,
      success: function(data) {
        var array = data.split(',')
        $(".form-check").append("<label><b>Seleccione las variables a utilizar</b></label><br>");
        for (let value of array) {
          var tmp = `
          <input class="form-check-input" type="checkbox" value=1 id="` + value + `" name=` + value + `>
          <label class="form-check-label" for="` + value + `">
            ` + value + `
          </label><br>
          `
          $(".form-check").append(tmp);
        }
        loadingFunct()
        change()
      }
    });
    return false;
    }
    
    $(function() {
        $('#form-file').submit(upload);
    });
});