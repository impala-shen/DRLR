import socket
import selectors
import types
import threading
import time
from typing import Tuple, Union, List

FileObj = Union[int, socket.socket]

class TCPOutputServer(threading.Thread):
    def __init__(self, port: int = 4242) -> None:
        super().__init__()
        self._running = False
        self._port = port
        self._clientList: List[socket.socket] = []
        self._cmd_arm = 0
        self._cmd_bucket_tilt = 0
        self._cmd_bucket = 0
        self._Safety_Flag = False
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._running:
            return
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind(("127.0.0.1", self._port))
        server_socket.listen()
        print("Started listening on port " + str(self._port))
        server_socket.setblocking(False)
        selector = selectors.DefaultSelector()
        selector.register(server_socket, selectors.EVENT_READ, data=None)

        self._serverSocket = server_socket
        self._selector = selector

        self._running = True
        super().start()

    def run(self) -> None:
        while self._running:
            try:
                events = self._selector.select(0.5)
                for key, mask in events:
                    if key.data is None:
                        self.AcceptConnection(key.fileobj)
                    else:
                        self.HandleMessage(key, mask)

                # Control send frequency by adding a delay
                time.sleep(0.1)  # 0.1 seconds = 10 Hz sending frequency
            except Exception as e:
                print(e)
        for c in self._clientList: c.close() # Disconnect clients
        self._selector.close()
        self._serverSocket.close()

    def stop(self) -> None:
        self._running = False
        self.join()

    def send_to_clients(self, message: str) -> None:
        # TODO: send to clients should store messages in a queue, which can then be sent to the
        # clients without blocking.
        with self._lock:
            for c in self._clientList:
                c.send(message.encode())

    def trans_to_bytes(self, value: float) -> List[str]:
        int_value = int(value * 10000)
        byte1 = (int_value >> 8) & 0xFF
        byte2 = int_value & 0xFF
        return [format(byte2, '02x'), format(byte1, '02x')]

    def degree_to_decimals(self, value: float) -> int:
        d = int(value)
        md = (value - d) * 60
        deg = d * 100
        return deg + md

    def SendIMUData(self, sensor_id: int, x: float, y: float, z: float, w: float) -> int:
        w_b = self.trans_to_bytes(w)
        x_b = self.trans_to_bytes(x)
        y_b = self.trans_to_bytes(y)
        z_b = self.trans_to_bytes(z)
        message = "CAN,0," + str(sensor_id) + ",8," + w_b[0] + w_b[1] + x_b[0] + x_b[1] + y_b[0] + y_b[1] + z_b[0] + z_b[1]
        try:
            self.send_to_clients(self.ToNMEAFormat(message))
            # print(message)
        except:
            return 0


    def send_imu_data(self, sensor_id, x, y, z, w):
        w_b = self.trans_to_bytes(w)
        x_b = self.trans_to_bytes(x)
        y_b = self.trans_to_bytes(y)
        z_b = self.trans_to_bytes(z)
        message = f"CAN,0,{sensor_id},0,{w_b[0]:02X}{w_b[1]:02X}{x_b[0]:02X}{x_b[1]:02X}{y_b[0]:02X}{y_b[1]:02X}{z_b[0]:02X}{z_b[1]:02X}"
        try:
            formatted_message = self.ToNMEAFormat(message)
            self.send_to_clients(self.ToNMEAFormat(message))
            # print(message)
            # # Simulate CAN message sending
            # bus = can.interface.Bus(channel='can0', bustype='socketcan')
            # data = [w_b[0], w_b[1], x_b[0], x_b[1], y_b[0], y_b[1], z_b[0], z_b[1]]
            # msg = can.Message(arbitration_id=sensor_id, data=data, is_extended_id=True)
            # bus.send(msg)
            # print(f"Sent CAN message: {msg}")
        except Exception as e:
            print(f"Error sending message: {e}")

    def SendMH4Data(self, sensor_id: int, a: float, b: float, c: float, d: float) -> int:
        message = "CAN,0," + str(sensor_id) + ",5," + a + b + c + d + "00"
        try:
            self.send_to_clients(self.ToNMEAFormat(message))
        except:
            return 0

    def SendGGAMessage(self, sensorID, time, la, lo, al):
        message = "GPGGA,111956.00," + format(self.degree_to_decimals(la), '.15f') + "," + "N" + "," + format(self.degree_to_decimals(lo), '.15f') + "," + "E,4,13,0.9,0,M," + str(al) + ",M,01,0000"
        self.send_to_clients(self.ToNMEAFormat(message))


    def SendHDTMessage(self, sensorID, heading):
        message = "GPHDT," + str(heading) + ",T*00"
        self.send_to_clients(self.ToNMEAFormat(message))

    def ToNMEAFormat(self,message):
        msg = '$' + message + '*'
        crc = 0
        for c in msg:
            crc ^= ord(c)
            extra = '' if crc > 15 else '0'
        # print (hex(crc).split('x')[1].upper())
        return msg + hex(crc).split('x')[1].upper() + "\n"

    def RPDToControl(self):
        data = self.ReadFromClients()
        can_id = data[0]

    def s16(self, value):
        return -(value & 0x8000) | (value & 0x7fff)

    def AcceptConnection(self, client_socket: FileObj) -> None:
        with self._lock:
            client, address = client_socket.accept()
            print(f"Accepted connection from {address}")
            client.setblocking(False)
            data = types.SimpleNamespace(addr=address, inb=b"", outb=b"")
            events = selectors.EVENT_READ | selectors.EVENT_WRITE
            self._selector.register(client, events, data=data)
            self._clientList.append(client)

    def CloseConnection(self, addr: int, client_socket: FileObj) -> None:
        with self._lock:
            print(f"Closing connection to {addr}")
            try:
                self._clientList.remove(client_socket)
            except:
                print("Already removed")
            self._selector.unregister(client_socket)
            client_socket.close()

    def HandleMessage(self, key: selectors.SelectorKey, mask):
        client = key.fileobj
        data = key.data
        if mask & selectors.EVENT_READ:
            recv_data = client.recv(1024)
            if not recv_data:
                self.CloseConnection(data.addr, client)
            else:
                data.outb += recv_data
                data_dec = recv_data.decode("utf-8")
                # print("Receive data:", data_dec)
                data_l = list(data_dec.split(","))
                # print("list data:", data_l)
                cmd_msg = data_l[2]
                # print("cmd message:", cmd_msg)
                data_messg = data_l[4]
                # print("data message:", data_messg)
                data_i = list(data_messg.split("*"))[0]
                if "208" in cmd_msg:
                    with self._lock:
                        self._Safety_Flag = True
                        self._cmd_arm = self.s16(int(swap_bytes(data_i[0:4]), base=16))
                        self._cmd_bucket_tilt = self.s16(int(swap_bytes(data_i[4:8]), base=16))
                        self._cmd_bucket = self.s16(int(swap_bytes(data_i[8:12]), base=16))
                else:
                    self._Safety_Flag = False
        elif mask & selectors.EVENT_WRITE and data.outb:
            sent = client.send(data.outb)
            data.outb = data.outb[sent:]

    def GetControlComd(self) -> Tuple[int, int, int, bool]:
        """ Last control command.

        Returns:
            Tuple[int, int, int, bool]: Arm, bucket tilt and bucket power levels, and safety flag
        """
        with self._lock:
            return self._cmd_arm, self._cmd_bucket_tilt, self._cmd_bucket, self._Safety_Flag
        # data = list(data_dec)
        # int(data, base=16)
        # while True:
            # data = data_dec.Data
            # print("Receive data:", data_dec)
        # self.HandleMessage()
        # int(data_dec, base = 16)

def swap_bytes(arr):
    return ''.join([i for s in [arr[2:4], arr[0:2]] for i in s])
