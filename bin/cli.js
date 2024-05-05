#!/usr/bin/env node
const path = require('path');
const program = require('commander');

program.command('open')
  .alias('o')
  .description('Open a project in browser.')
  .arguments('<name>')
  .action((name) => {
    // require('../lib/open')(name)
    require(path.join('..', 'lib', 'open'))(name)
})

program.parse(process.argv);