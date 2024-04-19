// Generated by CoffeeScript 1.12.7
(function() {
  var Snowflake;

  Snowflake = (function() {
    function Snowflake(canvas) {
      var side;
      this.canvas = canvas;
      side = Math.min(this.canvas.scrollWidth, this.canvas.scrollHeight);
      this.sx = side;
      this.sy = side;
      this.padding = 5;
      this.raphael = Raphael(this.canvas, side, side);
    }

    Snowflake.prototype.redraw = function(lod, useTriangles) {
      var s3, sx, sy;
      if (useTriangles == null) {
        useTriangles = false;
      }
      this.raphael.clear();
      this.pathCommand = null;
      this.statsClear();
      sx = this.sx - 2 * this.padding;
      sy = this.sy - 2 * this.padding;
      s3 = Math.sqrt(3);
      this.sideLength = sx * s3 / 2;
      if (useTriangles) {
        this.triangleStart(this.padding + sx / 2, this.padding);
        this.triangle(sx * -s3 / 4, sy * 3 / 4, sx * s3 / 2, 0, lod);
      } else {
        this.pathStart(this.padding + sx / 2, this.padding);
      }
      this.edge(lod, sx * -s3 / 4, sy * 3 / 4);
      this.edge(lod, sx * s3 / 2, 0);
      this.edge(lod, sx * -s3 / 4, sy * -3 / 4);
      if (this.pathCommand) {
        return this.pathEnd();
      }
    };

    Snowflake.prototype.edge = function(lod, xDelta, yDelta) {
      if (lod === 0) {
        if (this.pathCommand) {
          return this.pathLine(xDelta, yDelta);
        } else {
          return this.triangleMove(xDelta, yDelta);
        }
      } else {
        this.edge(lod - 1, xDelta / 3, yDelta / 3);
        if (!this.pathCommand) {
          this.triangle(xDelta / 6 - yDelta * Math.sqrt(3) / 6, yDelta / 6 + xDelta * Math.sqrt(3) / 6, xDelta / 6 + yDelta * Math.sqrt(3) / 6, yDelta / 6 - xDelta * Math.sqrt(3) / 6);
        }
        this.edge(lod - 1, xDelta / 6 - yDelta * Math.sqrt(3) / 6, yDelta / 6 + xDelta * Math.sqrt(3) / 6);
        this.edge(lod - 1, xDelta / 6 + yDelta * Math.sqrt(3) / 6, yDelta / 6 - xDelta * Math.sqrt(3) / 6);
        return this.edge(lod - 1, xDelta / 3, yDelta / 3);
      }
    };

    Snowflake.prototype.pathStart = function(xStart, yStart) {
      return this.pathCommand = ['M', xStart, ',', yStart, 'l'];
    };

    Snowflake.prototype.pathLine = function(xDelta, yDelta) {
      this.pathCommand.push(xDelta, ',', yDelta, ' ');
      this.lineCount += 1;
      xDelta /= this.sideLength;
      yDelta /= this.sideLength;
      return this.lineLength += Math.sqrt(xDelta * xDelta + yDelta * yDelta);
    };

    Snowflake.prototype.pathEnd = function() {
      this.pathCommand.push('c');
      return this.raphael.path(this.pathCommand.join(''));
    };

    Snowflake.prototype.triangleStart = function(xStart, yStart) {
      this.triX = xStart;
      return this.triY = yStart;
    };

    Snowflake.prototype.triangleMove = function(xDelta, yDelta) {
      this.triX += xDelta;
      return this.triY += yDelta;
    };

    Snowflake.prototype.triangle = function(edge1X, edge1Y, edge2X, edge2Y, lod) {
      var pathCommand;
      pathCommand = ['M', this.triX, ',', this.triY, 'l', edge1X, ',', edge1Y, 'l', edge2X, ',', edge2Y, 'Z'];
      this.raphael.path(pathCommand.join('')).attr('fill', '#e0e0e0').attr('stroke', '#202020');
      this.triangleCount += 1;
      edge1X /= this.sideLength;
      edge1Y /= this.sideLength;
      return this.triangleArea += Math.sqrt(3) * (edge1X * edge1X + edge1Y * edge1Y) / 4;
    };

    Snowflake.prototype.statsClear = function() {
      this.lineCount = 0;
      this.lineLength = 0;
      this.triangleCount = 0;
      return this.triangleArea = 0;
    };

    return Snowflake;

  })();

  window.onload = function() {
    var event, i, input, j, len, len1, lineCountSpan, lineLengthSpan, lodInput, redraw, ref, ref1, render2DInput, render3DInput, snowFlake, triangleAreaSpan, triangleCountSpan;
    snowFlake = new Snowflake(document.getElementById('canvas'));
    lodInput = document.getElementById('lod');
    render2DInput = document.getElementById('render-2d');
    render3DInput = document.getElementById('render-3d');
    lineCountSpan = document.getElementById('line-count');
    lineLengthSpan = document.getElementById('line-length');
    triangleCountSpan = document.getElementById('triangle-count');
    triangleAreaSpan = document.getElementById('triangle-area');
    redraw = function() {
      snowFlake.redraw(parseInt(lodInput.value), render3DInput.checked);
      lineCountSpan.innerHTML = snowFlake.lineCount;
      lineLengthSpan.innerHTML = snowFlake.lineLength;
      triangleCountSpan.innerHTML = snowFlake.triangleCount;
      return triangleAreaSpan.innerHTML = snowFlake.triangleArea;
    };
    ref = [lodInput, render2DInput, render3DInput];
    for (i = 0, len = ref.length; i < len; i++) {
      input = ref[i];
      ref1 = ['change', 'click', 'keyup'];
      for (j = 0, len1 = ref1.length; j < len1; j++) {
        event = ref1[j];
        input.addEventListener(event, redraw, false);
      }
    }
    return redraw();
  };

}).call(this);