//To-do investigate using namespaces
var WebSocketServer = require('websocket').server;
var fs = require('fs');
var net = require('net');
var express = require('express');
var app = express();
var http = require('http').Server(app);

var unix_socket = '/var/run/rTreeDemo';
// var io = require('socket.io')(http);


  global_counter = 0;
  all_active_connections = {};

app.use('/assets', express.static(__dirname +'/assets'));
app.use('/bower_components', express.static(__dirname +'/bower_components'));

app.get('/', function (req, res){
   res.sendFile(__dirname + '/index.html');
});

http.listen(3000, function(){
  console.log('listening on *:3000');
});

wsServer = new WebSocketServer({
    httpServer: http,
    // You should not use autoAcceptConnections for production 
    // applications, as it defeats all standard cross-origin protection 
    // facilities built into the protocol and the browser.  You should 
    // *always* verify the connection's origin and decide whether or not 
    // to accept it. 
    autoAcceptConnections: false
});

function originIsAllowed(origin) {
  // put logic here to detect whether the specified origin is allowed. 
  return true;
}

wsServer.on('request', function(request) {
    if (!originIsAllowed(request.origin)) {
      // Make sure we only accept requests from an allowed origin 
      request.reject();
      console.log((new Date()) + ' Connection from origin ' + request.origin + ' rejected.');
      return;
    }
    
    var connection = request.accept('echo-protocol', request.origin);

    console.log((new Date()) + ' Connection accepted.');

     var id = global_counter++;
     all_active_connections[id] = connection;
     connection.id = id; 
    connection.on('message', function(message) {
        if (message.type === 'utf8') {
            console.log('Received Message: ' + message.utf8Data);
            connection.sendUTF(message.utf8Data);
             for (conn in all_active_connections)
                 all_active_connections[conn].sendUTF(message.utf8Data);
        }
        else if (message.type === 'binary') {
            console.log('Received Binary Message of ' + message.binaryData.length + ' bytes');
            connection.sendBytes(message.binaryData);
        }
    });
    connection.on('close', function(reasonCode, description) {
        console.log((new Date()) + ' Peer ' + connection.remoteAddress + ' disconnected.');
    });
});

var unixServer = net.createServer(function(client) {
    // Do something with the client connection
    client.setEncoding('utf8');
    client.on('data', function(chunk) {
    console.log('got %d bytes of data %s', chunk.length, chunk.toString());
   for (conn in all_active_connections)
                 all_active_connections[conn].send(chunk.toString());
  });
});

unixServer.on('close', function(){
  console.log("Close Unix Server");
});

unixServer.listen(unix_socket);


process.stdin.resume();//so the program will not close instantly

function exitHandler(options, err) {
  unixServer.close();
    if (options.cleanup) console.log('clean');
    if (err) console.log(err.stack);
    if (options.exit) process.exit();
}

//do something when app is closing
process.on('exit', exitHandler.bind(null,{cleanup:true}));

//catches ctrl+c event
process.on('SIGINT', exitHandler.bind(null, {exit:true}));

//catches uncaught exceptions
process.on('uncaughtException', exitHandler.bind(null, {exit:true}));
