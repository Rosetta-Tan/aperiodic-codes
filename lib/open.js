const exec = require('child_process').exec;
const { chalkGreen } = require('../utils');

// open specified subproject with port 5501
function open(name) {
  exec(`parcel static/${name}/index.html --no-cache -p 5501 --open`, err => {
    if (err) throw err
  })
  chalkGreen(`Open ${name}`)
}

module.exports = (...args) => open(...args)