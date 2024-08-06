const textarea = document.getElementById('message');


textarea.value = '';


const btn = document.getElementById('btn');

btn.addEventListener('click', function handleClick() {

  console.log(textarea.value);

  
  textarea.value = '';
});