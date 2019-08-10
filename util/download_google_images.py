from google_images_download import google_images_download

arguments = {"keywords":"shoes,shoes baseball,red shoes,nike shoes,adidas shoes,formal shoes,vans shoes,tennis shoes,shoes for women,exercise shoes,basketball shoes,black shoes,white shoes,leather shoes,shoes for boys,shoes for men,walking shoes,running shoes","limit":100,"output_directory":"/home/algolaptop8/ws/deeplearn/data/caps and shoes extended/shoes","no_directory" : True}
#arguments = {"keywords":"cap,cap baseball,red cap,blue cap,pink cap,cricket cap,golf cap,tennis cap,badminton cap,quality cap,basketball cap,black cap,white cap,round cap,cap for boys,cap for men,cap for women","limit":100,"output_directory":"/home/algolaptop8/ws/deeplearn/data/caps and shoes extended/caps","no_directory" : True}
response = google_images_download.googleimagesdownload()
absolute_image_paths = response.download(arguments)


