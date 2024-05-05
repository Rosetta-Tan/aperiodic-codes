const path = require('path');
const exec = require('child_process').exec;
const { chalkGreen } = require(path.join('..', 'utils'));

// open specified subproject with port 5501
function open(name) {
  const filePath = path.join('static', name, 'index.html');
  exec(`parcel ${filePath} --no-cache -p 5501 --open`, err => {
    if (err) throw err
  })
  chalkGreen(`Open ${name}`)
}

module.exports = (...args) => open(...args)