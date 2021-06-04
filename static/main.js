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
    axios.post('/enable/blush', {
        r_value: 74,
        g_value: 136,
        b_value: 237,
        intensity: 0.3
    })
        .then(res => {
            console.log(res)
        })
        .catch(err => {
            console.log(err)
        })
}

let eyeshadow = () => {
    axios.post('/enable/eyeshadow', {
        r_value: 74,
        g_value: 136,
        b_value: 237,
        intensity: 0.3
    })
        .then(res => {
            console.log(res)
        })
        .catch(err => {
            console.log(err)
        })
}

let lipstick = () => {
    axios.post('/enable/lipstick', {
        r_value: 74,
        g_value: 136,
        b_value: 237,
        intensity: 0.3
    })
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