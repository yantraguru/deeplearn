from google_images_download import google_images_download

#arguments = {"keywords":"shoes,shoes baseball,red shoes,nike shoes,adidas shoes,formal shoes,vans shoes,tennis shoes,shoes for women,exercise shoes,basketball shoes,black shoes,white shoes,leather shoes,shoes for boys,shoes for men,walking shoes,running shoes","limit":100,"output_directory":"/home/algolaptop8/ws/deeplearn/data/caps and shoes extended/shoes","no_directory" : True}
#arguments = {"keywords":"marathon shoes,elite shoes,kobe shoes,green shoes,mamba shoes,low cut shoes,orange shoes,green shoes,oxford shoes,formal shoes loafer,red tape shoes,bata shoes,designer shoes, suede formal shoes","limit":100,"output_directory":"/home/algolaptop8/ws/deeplearn/data/caps and shoes extended/shoes","no_directory" : True}
#arguments = {"keywords":"cap,cap baseball,red cap,blue cap,pink cap,cricket cap,golf cap,tennis cap,badminton cap,quality cap,basketball cap,black cap,white cap,round cap,cap for boys,cap for men,cap for women","limit":100,"output_directory":"/home/algolaptop8/ws/deeplearn/data/caps and shoes extended/caps","no_directory" : True}
#arguments = {"keywords":"stylish caps,team caps,snapback hat,trucker hats,embroidery caps,metal plate cap,walmart caps,kids caps,love caps, disney caps,avenger cap,european cap,british cap,cap modern","limit":50,"output_directory":"/home/algolaptop8/ws/deeplearn/data/caps and shoes extended/caps","no_directory" : True}

arguments = {"keywords":"stylish hat,hat,formal hat,panama hat,wool hat,winter cap,game cap,traning caps,flag cap", "limit":100,"output_directory":"/home/algolaptop8/ws/deeplearn/data/caps_and_shoes_extended_2/caps","no_directory" : True}


response = google_images_download.googleimagesdownload()
absolute_image_paths = response.download(arguments)


