sudo yum update
curl -O https://bootstrap.pypa.io/get-pip.py
python get-pip.py
pip install --upgrade pip
pip install awscli --upgrade --user
sudo yum install docker -y
sudo service docker start
sudo tar -cv -C /mnt/ . | sudo docker import - YOUR-IMAGE-NAME

0</dev/null 1> log.out 2>log.err &
