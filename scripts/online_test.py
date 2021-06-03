import asyncio
import pathlib

import websockets
import socket
import pyrebase
import ssl
import subprocess

firebaseConfig = {
  "apiKey": "AIzaSyATjdo9-LnDVRgxfeuQaOopO4EIFmTXEik",
  "authDomain": "bioradar.firebaseapp.com",
  "databaseURL": "https://bioradar-default-rtdb.europe-west1.firebasedatabase.app",
  "projectId": "bioradar",
  "storageBucket": "bioradar.appspot.com",
  "messagingSenderId": "857754767607",
  "appId": "1:857754767607:web:78f269b3f4da42c60343f2",
  "measurementId": "G-0SC2CY2RDM"
}

def main():
    IP_List = socket.gethostbyname_ex(socket.gethostname())[-1]
    print(IP_List[len(IP_List)-1])
    print(IP_List)
    #IPAddr = IP_List[len(IP_List)-1]
    IPAddr = '192.168.100.223'
    print("RPi IP Address: " + IPAddr)
    print("Waiting for online commands...")
    port = 30000

    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    localhost_pem = pathlib.Path(__file__).with_name("mycertificate.pem")
    ssl_context.load_cert_chain(localhost_pem)

    firebase = pyrebase.initialize_app(firebaseConfig)
    db=firebase.database()

    Connection_data={"IP":IPAddr, "Port": port}
    db.child('Connection_Data').set(Connection_data)
    
    asyncio.get_event_loop().run_until_complete(websockets.serve(echo, IPAddr, port, ssl=ssl_context))
    asyncio.get_event_loop().run_forever()

async def echo(websocket, path):
    async for message in websocket:
        print(message)
        if message == 'StartHR':
            #send message to tell webUI when to deactivate button
            await websocket.send('Processing')
            
            print("Starting RR/HR acquisition...\n")
            
            # START
            
            #await asyncio.sleep(2)
            subprocess.run('python3 scripts/run_rr_hr_online.py -u', shell=True)
            
            # END
            

            # to stop loop, use this line of code
            # asyncio.get_event_loop().stop()

            # send message to tell webUI when to reactivate button
            await websocket.send('Finished')
        elif message == 'StartHRV':
            await websocket.send('Processing')
            print("Starting HRV/HR acquisition...\n")
            subprocess.run('python3 scripts/run_hrv_hr_online.py -u', shell=True)
            await websocket.send('Finished')

main()