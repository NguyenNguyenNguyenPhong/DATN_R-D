let imageArray1 = {{ uploaded_image_paths_1 | tojson }};
let imageArray2 = {{ uploaded_image_paths_2 | tojson }};
let file_name = ""
let name = "{{ name }}"

function updateSlides(paths1, paths2) {
  if (paths1.length > 0) {
    document.getElementById('slide-image-1').src = paths1[Math.floor(imageArray1.length / 2)];
  }

  if (paths2.length > 0) {
    document.getElementById('slide-image-2').src = paths2[Math.floor(imageArray2.length / 2)];
  }
}

function scrollSlideContainer1(e) {
  // Check scroll direction and update current slide
  currentSlide1 += (e.deltaY > 0) ? 1 : -1;
  currentSlide1 = Math.max(0, Math.min(currentSlide1, imageArray1.length - 1));
  updateSlide1();
}

function scrollSlideContainer2(e) {
  // Check scroll direction and update current slide
  currentSlide2 += (e.deltaY > 0) ? 1 : -1;
  currentSlide2 = Math.max(0, Math.min(currentSlide2, imageArray2.length - 1));
  updateSlide2();
}

// Function to fetch updated image paths using AJAX
function fetchUpdatedImagePaths() {
  const nameParam = encodeURIComponent(name);
  fetch(`/get_image_paths/${nameParam}`)
    .then(response => response.json())
    .then(data => {
      // Update imageArray1 and imageArray2 with the new paths
      imageArray1 = data.image_paths_1;
      imageArray2 = data.image_paths_2;
      updateSlides(imageArray1, imageArray2);
    });
}


fetchUpdatedImagePaths();

const slideImage1 = document.getElementById('slide-image-1');
const backButton1 = document.getElementById('backButton1');
const nextButton1 = document.getElementById('nextButton1');
let currentSlide1 = Math.floor(imageArray1.length / 2);

backButton1.addEventListener('click', () => {
  currentSlide1 = (currentSlide1 - 1 + imageArray1.length) % imageArray1.length;
  updateSlide1();
});

nextButton1.addEventListener('click', () => {
  currentSlide1 = (currentSlide1 + 1) % imageArray1.length;
  updateSlide1();
});

function updateSlide1() {
  if (imageArray1.length > 0) {
    slideImage1.src = imageArray1[currentSlide1];
  }
}

updateSlide1(); // Load the first image

//list 2
const slideImage2 = document.getElementById('slide-image-2');
const backButton2 = document.getElementById('backButton2');
const nextButton2 = document.getElementById('nextButton2');
let currentSlide2 = Math.floor(imageArray2.length / 2);

backButton2.addEventListener('click', () => {
  currentSlide2 = (currentSlide2 - 1 + imageArray2.length) % imageArray2.length;
  updateSlide2();
});

nextButton2.addEventListener('click', () => {
  currentSlide2 = (currentSlide2 + 1) % imageArray2.length;
  updateSlide2();
});

function updateSlide2() {
  if (imageArray2.length > 0) {
    slideImage2.src = imageArray2[currentSlide2];
  }
}
const segmentButton = document.getElementById('segmentButton');
const segmentButtonContent = document.getElementById('segmentButtonContent');


function setSegmentButtonState(isEnabled) {
  if (isEnabled) {
    segmentButton.disabled = false;
    segmentButtonContent.querySelector('.fa-circle-notch').style.display = 'none';
    segmentButtonContent.querySelector('.fa-check').style.display = 'inline-block';
    segmentButtonContent.querySelector('span').innerText = 'Show Result';
  } else {
    segmentButton.disabled = true;
    segmentButtonContent.querySelector('.fa-circle-notch').style.display = 'inline-block';
    segmentButtonContent.querySelector('.fa-check').style.display = 'none';
    segmentButtonContent.querySelector('span').innerText = 'Processing.';
  }
}

function fetchSegmentedFilename() {
  fetch('/get_segmented_filename')
    .then(response => response.json())
    .then(data => {
      if (data.segmented_filename) {
        file_name = data.segmented_filename
        console.log(file_name)
        // If the segmented filename exists, enable the Segment button
        setSegmentButtonState(true);
      }
    });
}

segmentButton.addEventListener('click', () => {
  if (file_name !== '') {
    const filenameParam = encodeURIComponent(file_name);
    window.location.href = `/segmentation_display/${filenameParam}`;
  }
});

const slideContainer1 = document.getElementById('slide-container-1');
const slideContainer2 = document.getElementById('slide-container-2');
slideContainer1.addEventListener('wheel', scrollSlideContainer1);
slideContainer2.addEventListener('wheel', scrollSlideContainer2);

updateSlide2(); // Load the first image

setInterval(fetchSegmentedFilename, 2000);