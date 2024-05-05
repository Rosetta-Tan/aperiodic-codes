const path = require('path');
const chalk = require('chalk');

// print message
const chalkGreen = msg => console.log(chalk.green(msg))
const chalkRed = msg => console.log(chalk.red(msg))

// get path (absolute path under root directory)
const joinPath = filePath => path.join(__dirname, filePath)

module.exports = {
  chalkGreen,
  chalkRed,
  joinPath
}