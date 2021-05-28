let openCam = () => {
    axios.post('/open-cam')
        .then(res => {
            console.log(res)
            // $( "#container" ).append(`<img id="vid-feed" src="/video-feed" width="100%" alt="video-feed"></img>`);
        })
        .catch(err => {
            console.log(err)
        })
}

let closeCam = () => {
    // $( "#vid-feed" ).remove();
    axios.post('/close-cam')
        .then(res => {
            console.log(res)
        })
        .catch(err => {
            console.log(err)
        })
}

let blush = () => {
    axios.get('/blush')
        .then(res => {
            console.log(res)
        })
        .catch(err => {
            console.log(err)
        })
}

let eyeshadow = () => {
    axios.get('/eyeshadow')
        .then(res => {
            console.log(res)
        })
        .catch(err => {
            console.log(err)
        })
}

let lipstick = () => {
    axios.get('/lipstick')
        .then(res => {
            console.log(res)
        })
        .catch(err => {
            console.log(err)
        })
}

setInterval(() => {
    $('#vid-feed').attr("src", "/video-feed?" + new Date().getTime())
}, 1000);

// let toggle_feed =