<?php
   $file_path = "/home/ubuntu/Project/images/";
   $file_path = $file_path.basename( $_FILES['file']['name']);
   $filename = $_FILES['file']['name'];
 
   if(move_uploaded_file($_FILES['file']['tmp_name'], $file_path)) {
	    $result = array("result" => "success");
	    
    } else{
        $result = array("result" => "error");
    }

	//if the filename has a 'zoom', execute resolution.py
   if(strpos($filename, "zoom") !== false){
 	shell_exec("/home/ubuntu/anaconda3/bin/python resolution.py original_zoom.jpg"); 
  }
   else{ //if not, execute outfocus.py
  	shell_exec("/home/ubuntu/anaconda3/bin/python outfocus.py original.jpg");

  }
?>
