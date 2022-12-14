(function() {

  var questions;
  var mode=0;
  var custom_message = ">>> Write your own question <<<";

  $( window ).init(function(){
		load();
	});

	function load(){

    loadSampleQuestions();

    $('.editOption').keyup(function () {
      var editText = $('.editOption').val();
      $('editable').val(editText);
      $('editable').html(editText);
    });

    $('.editOption').on('click', function () {
      $('#answer').html('')
    });

    $(".run").click(loadAnswer);

    $('.editOption').keyup(function(event) {
      if (event.keyCode === 13) {
        event.preventDefault();
        if (event.stopPropagation!=undefined) {
          event.stopPropagation();
        }
        loadAnswer();
      }
    })


    $('.mode').click(function() {
      mode = parseInt($('.mode:checked').val());
      $('#answer').html('');
      if (mode === 0) {
        $('#select-question').show();
        $('#write-question').hide();
        $('#refresh').show();
      } else {
        $('#select-question').hide();
        $('#write-question').show();
        $('#refresh').hide();
      }
    });

    /* Label Tooltip */
    $('.mode').mouseover(function(event){
      $('#mode-tooltip').removeClass('tooltip-hidden').addClass('tooltip-visible');
      if (parseInt(event.target.value)===0)
        $('#mode-tooltip').html("You can see example questions from OpenBookQA, ARC, AI2Science, CommonsenseQA, QASC, Physical IQA, Social IQA, Winogrande.");
      else
        $('#mode-tooltip').html("You can write your own questions.");
    });
    $('.mode').mouseout(function(){
      $('#mode-tooltip').removeClass('tooltip-visible').addClass('tooltip-hidden');
      $('#mode-tooltip').html(""); // this prevents newline
    });
    $('#k-div').mouseover(function(event){
      $('#k-tooltip').removeClass('tooltip-hidden').addClass('tooltip-visible');
      $('#k-tooltip').html("Number of answers to be returned (1--100).");
    });
    $('#k-div').mouseout(function(){
      $('#k-tooltip').removeClass('tooltip-visible').addClass('tooltip-hidden');
      $('#k-tooltip').html(""); // this prevents newline
    });

    /* Description Button */
    $('#description-button').click(function(){
      if ($('#description-button').html() === 'Show Me Details!') {
        $('#description').show();
        $('#description-button').html('Hide Details!');
      } else {
        $('#description').hide();
        $('#description-button').html('Show Me Details!');
      }
    })

  }

  function loadSampleQuestions() {
    sendAjax("/select", {}, (data) => {
      questions = data.questions;
      var dropdown = document.getElementById("question");
      for(var i=0; i<questions.length; i++){
				var opt = document.createElement("option");
				opt.value = parseInt(i);
        opt.id = "question-option-"+parseInt(i);
				opt.innerHTML = questions[i];
        dropdown.appendChild(opt);
      }
    })
  }

  function loadAnswer(){
    var question_text = $('select#question option:selected').html();
    if (mode === 1) {
      question_text = $('.editOption').val();
      if (!(question_text.replace(/\s/g, '').length)) {
        alert('Please enter a non-empty question.');
        return;
      }
    }
		document.getElementById("answer").innerHTML = "";
		document.getElementById("loading").style.display = "block";
		var data = {
      'question': question_text
    };
		sendAjax("/submit", data, (result) => {
			document.getElementById("loading").style.display = "none";
      var answer_field = document.getElementById('answer');
        var header = `<b>Knowledges</b>
          <span class='footnote'><b>QA model answer without knowledge</b>: ` + result["knowless_pred"] + `</span>
          <span class='footnote'><b>QA model answer with knowledge</b>: ` + result["knowful_pred"] + `</span>`;
        var content = "";
        for (var i = 0; i < result["knowledges"].length; i++) {
          var knowledge = result["knowledges"][i];
          if (knowledge === "") {
            continue;
          }
          if (knowledge === result["selected_knowledge"]) {
            content += "<b>" + knowledge + "</b><br>";
          } else {
            content += knowledge + "<br>";
          }
        }
        answer_field.appendChild(getPanel(header, content));
    });
	}
  function sendAjax(url, data, handle){
		$.getJSON(url, data, function(response){
			handle(response.result);
		});
	}

	function getPanel(heading_text, context_text){
		var div = document.createElement('div');
		div.className = "panel panel-default";
		var heading = document.createElement('div');
		heading.className = "panel-heading";
		heading.innerHTML = heading_text;
		var context = document.createElement('div');
		context.className = "panel-body";
		context.innerHTML = context_text;
		div.appendChild(heading);
		div.appendChild(context);
		return div
	}

  function getPanel2(heading_text, context_text, footer_text){
		var div = document.createElement('div');
		div.className = "panel panel-default";
		var heading = document.createElement('div');
    heading.className = "panel-heading my-heading";
    heading.style.width = "150px";
    heading.style.float = "left";
		heading.innerHTML = heading_text;
    var context = document.createElement('div');
    context.className = "panel-body";
    //context.style.float = "left";
		context.innerHTML = context_text;
    var footer = document.createElement('div');
    footer.className = "panel-heading";
    footer.style.width = "150px";
    footer.style.float = "right";
		footer.innerHTML = footer_text;


    div.appendChild(heading);
    div.appendChild(footer);
    div.appendChild(context);

    //heading.style.height = context.height;
    //footer.style.height = context.height;

    return div
	}


})();



