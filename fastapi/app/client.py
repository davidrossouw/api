
# Requests library is what you need. You can install with pip install requests.
# http://docs.python-requests.org/en/latest/user/quickstart/#post-a-multipart-encoded-file

url = 'http://httpbin.org/post'


def post_image(img_file, URL):
    """ post image and return the response """
    img = open(img_file, 'rb').read()
    response = requests.post(URL, data=img, headers=headers)
    return response

post_image('lena.jpg', 'http://127.0.0.1:8000/test')