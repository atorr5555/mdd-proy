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
        $(".form-check").append("<label><b>Seleccione la variable a pronosticar</b></label><br>");
        for (let value of array) {
          var tmp = `
          <input class="form-check-input" type="radio" value=` + value + ` id="` + value + `" name="seleccion">
          <label class="form-check-label" for="` + value + `">
            ` + value + `
          </label><br>
          `
          $(".form-check").append(tmp);
        }
        $(".form-check").append("<label><b>Seleccione las variables predictoras</b></label><br>");
        for (let value of array) {
          var tmp = `
          <input class="form-check-input" type="checkbox" value=1 id="` + value + `" name=` + value + `>
          <label class="form-check-label" for="` + value + `">
            ` + value + `
          </label><br>
          `
          $(".form-check").append(tmp);
        }
        $(".form-check").append("<label><b>Configure los parámetros</b></label><br>");
        $(".form-check").append(`
        <label for="profundidad">Profundidad maxima</label>
        <input type="number" min="0" max="30" name="profundidad" id="profundidad"><br>
        <label for="division">Mínimo de muestras para dividir</label>
        <input type="number" min="0" max="30" name="division" id="division"><br>
        <label for="hoja">Mínimo de muestras en un nodo hoja</label>
        <input type="number" min="0" max="30" name="hoja" id="hoja"><br>
        `);
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