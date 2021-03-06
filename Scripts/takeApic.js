// Grab elements, create settings, etc.
var video = document.getElementById('videoScreen');

// Get access to the camera!
if(navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    // Not adding `{ audio: true }` since we only want video now
    navigator.mediaDevices.getUserMedia({ video: true }).then(function(stream) {
            video.src = window.URL.createObjectURL(stream);
            video.play();
    });
}

// Elements for taking the snapshot
var canvas = document.getElementById('canvas');
var context = canvas.getContext('2d');
var video = document.getElementById('videoScreen');

// Trigger photo take
document.getElementById("snap").addEventListener("click", function() {
        context.drawImage(video, 0, 0, 640, 480)
        var image = convertCanvasToImage(context);
});

function convertCanvasToImage(canvas) {
	var image = new Image();
        image.src = canvas.toDataURL("image/jpeg");
        localStorage.setItem("Images/image1", "image/jpeg");        
        return image;        
}