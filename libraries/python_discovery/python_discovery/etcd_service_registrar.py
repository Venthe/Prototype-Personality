import socket
import subprocess
import etcd3
import time
import os
import logging

class EtcdServiceRegistrar:
    def __init__(self, etcd_host='localhost', etcd_port=2379, ttl=60):
        self.etcd = etcd3.client(host=etcd_host, port=etcd_port)
        self.ttl = ttl
        self.service_id = None
        self.hostname = socket.gethostname()
        self.port = None
        self.health_endpoint = None
        logging.basicConfig(level=logging.INFO)

    def find_unused_port(self):
        """Find an available port."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(('', 0))
            self.port = sock.getsockname()[1]
        return self.port

    def start_application(self, command):
        """Start the application as a subprocess."""
        self.process = subprocess.Popen(command, shell=True)
        logging.info(f'Started application with PID {self.process.pid}')

    def register_service(self, service_id, health_endpoint):
        """Register service information with etcd."""
        self.service_id = service_id
        self.health_endpoint = health_endpoint
        self.etcd.put(service_id, f'{self.hostname}:{self.port}', lease=self.etcd.lease(self.ttl))
        logging.info(f'Registered service {service_id} at {self.hostname}:{self.port}')

    def health_check(self):
        """Perform a health check and update etcd if alive."""
        while True:
            # Implement actual health check logic here
            # For example, check if the service is responding
            if self.process.poll() is not None:
                logging.warning(f'Service {self.service_id} is not running. Deregistering...')
                self.deregister_service()
                break
            time.sleep(10)  # Adjust based on your health check frequency

    def deregister_service(self):
        """Deregister the service from etcd."""
        if self.service_id:
            self.etcd.delete(self.service_id)
            logging.info(f'Deregistered service {self.service_id}')
        
    def run(self, command, service_id, health_endpoint):
        """Start the service and handle registration."""
        self.find_unused_port()
        self.start_application(command)
        self.register_service(service_id, health_endpoint)
        self.health_check()

    def discover_service(self, service_id):
        """Discover a service by its ID."""
        service = self.etcd.get(service_id)
        if service:
            address = service[0].decode('utf-8')
            logging.info(f'Service {service_id} found at {address}')
            return address
        else:
            logging.warning(f'Service {service_id} not found')
            return None

    def watch_service(self, service_id, callback):
        """Watch a service for changes."""
        events_iterator, cancel = self.etcd.watch(service_id)
        for event in events_iterator:
            if event.events:
                new_value = event.events[0].kv.value.decode('utf-8')
                logging.info(f'Service {service_id} updated: {new_value}')
                callback(new_value)
            else:
                logging.warning(f'Service {service_id} deleted or not available.')
                callback(None)

if __name__ == "__main__":
    registrar = EtcdServiceRegistrar()
    registrar.run("your_application_command_here", "service_id", "http://your_health_endpoint")
