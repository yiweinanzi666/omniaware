import argparse, datetime, getpass, hashlib, json, pathlib, select, socket, threading, time
import paramiko

ROOT = pathlib.Path(__file__).resolve().parent
REMOTE = '/home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017'
KEY = pathlib.Path.home()/'.ssh/omniaware_node04_ed25519'

def connect(password=None):
    jump = paramiko.SSHClient(); jump.load_system_host_keys(); jump.load_host_keys(str(pathlib.Path.home()/'.ssh/known_hosts'))
    auth={'password':password} if password else {'key_filename':str(KEY)}
    jump.connect('172.30.0.6', username='user', **auth, look_for_keys=False, allow_agent=False, timeout=30)
    jump.get_transport().set_keepalive(15)
    channel = jump.get_transport().open_channel('direct-tcpip', ('node04',22), ('127.0.0.1',0))
    node = paramiko.SSHClient(); node.load_host_keys(str(pathlib.Path.home()/'.ssh/known_hosts'))
    node.connect('node04', username='user', **auth, sock=channel, look_for_keys=False, allow_agent=False, timeout=30)
    node.get_transport().set_keepalive(15)
    return jump,node

def proxy_client(channel):
    target=None
    try:
        buf=b''
        while b'\r\n\r\n' not in buf and len(buf)<32768:
            b=channel.recv(4096)
            if not b:return
            buf+=b
        line=buf.split(b'\r\n',1)[0].decode('ascii'); method, dest, _=line.split(' ')
        host,port=dest.rsplit(':',1); port=int(port)
        allowed=('pypi.org','pythonhosted.org','pytorch.org','github.com','githubusercontent.com','githubassets.com','hf-mirror.com','huggingface.co','hf.co','xethub.hf.co','google.com','googleusercontent.com','gstatic.com','anaconda.com','conda.io','nvidia.com')
        if method!='CONNECT' or port!=443 or not any(host==x or host.endswith('.'+x) for x in allowed):
            channel.sendall(b'HTTP/1.1 403 Forbidden\r\n\r\n');return
        target=socket.create_connection((host,port),timeout=30)
        channel.sendall(b'HTTP/1.1 200 Connection established\r\n\r\n')
        rest=buf.split(b'\r\n\r\n',1)[1]
        if rest:target.sendall(rest)
        while True:
            ready,_,_=select.select([channel,target],[],[],120)
            if not ready:break
            for s in ready:
                data=s.recv(262144)
                if not data:return
                (target if s is channel else channel).sendall(data)
    except Exception as e:
        print(json.dumps({'proxy_error':str(e)}),flush=True)
    finally:
        channel.close()
        if target:target.close()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['exec','upload','proxy']);ap.add_argument('value',nargs='?');ap.add_argument('dest',nargs='?');a=ap.parse_args()
    if a.action=='proxy':
        password=getpass.getpass('SSH password: ')
        while True:
            jump=node=None
            try:
                jump,node=connect(password)
                port=node.get_transport().request_port_forward('127.0.0.1',18761)
                print('PROXY_READY localhost:'+str(port),flush=True)
                while node.get_transport().is_active():
                    ch=node.get_transport().accept(30)
                    if ch:threading.Thread(target=proxy_client,args=(ch,),daemon=True).start()
                raise RuntimeError('SSH tunnel disconnected')
            except Exception as e:
                print(json.dumps({'relay_reconnecting':str(e),'time':datetime.datetime.now().isoformat()}),flush=True)
            finally:
                if node:node.close()
                if jump:jump.close()
            time.sleep(15)
    jump,node=connect()
    try:
        if a.action=='proxy':
            port=node.get_transport().request_port_forward('127.0.0.1',18761)
            print('PROXY_READY localhost:'+str(port),flush=True)
            while node.get_transport().is_active():
                ch=node.get_transport().accept(30)
                if ch:threading.Thread(target=proxy_client,args=(ch,),daemon=True).start()
            raise RuntimeError('SSH tunnel disconnected')
        if a.action=='upload':
            p=pathlib.Path(a.value);s=node.open_sftp()
            try:s.stat(a.dest)
            except FileNotFoundError:pass
            else:raise RuntimeError('Destination exists; refusing overwrite: '+a.dest)
            s.put(str(p),a.dest);print(json.dumps({'uploaded':a.dest,'bytes':p.stat().st_size,'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()}));return
        cmd=pathlib.Path(a.value).read_text(encoding='utf8')
        start=datetime.datetime.now(datetime.timezone.utc).isoformat(); stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        _,out,err=node.exec_command(cmd,timeout=240)
        stdout=out.read().decode();stderr=err.read().decode();rc=out.channel.recv_exit_status()
        rec={'command':cmd,'start':start,'end':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':rc,'stdout':stdout,'stderr':stderr}
        encoded=json.dumps(rec,ensure_ascii=False,indent=2)
        (ROOT/(stamp+'.json')).write_text(encoded,encoding='utf8')
        s=node.open_sftp()
        with s.open(REMOTE+'/'+stamp+'.json','x') as f:f.write(encoded.encode())
        print(encoded)
    finally:
        node.close();jump.close()
if __name__=='__main__':main()
