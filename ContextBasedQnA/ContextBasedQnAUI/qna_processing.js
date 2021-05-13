function getAnswer() {
  element = document.getElementById("answer-div")
  element.style.display ="none"

  element = document.getElementById("error-div")
  element.style.display ="none"

  element = document.getElementById("loading-symbol")
  element.style.visibility ="visible"

  element = document.getElementById("context-input")
  context = element.value
  element = document.getElementById("question-input")
  question = element.value
  input = {
    'context': context,
    'question':question
  }
  fetch('http://127.0.0.1:5000/qna/', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(input),
  })
    .then(response => response.json())
    .then(data => {
      console.log(data)
      if (data.status == 200) {
        console.log('Success:', data);
        renderAnswer(data);
      }
    })
    .catch((error) => {
      element = document.getElementById("error-div")
      element.style.display ="block"
      element = document.getElementById("loading-symbol")
      element.style.visibility ="hidden"

      console.error('Error:', error);
    });
}

function renderAnswer(_data) {
  console.log(_data.answer)
  document.getElementById("answer-output").innerHTML = _data.answer

  element = document.getElementById("answer-div")
  element.style.display ="block"

  element = document.getElementById("loading-symbol")
  element.style.visibility ="hidden"
}
