import os, errno

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: 
            raise

def output( msg ):
    outfile = open('OUTPUT','a')
    outfile.write( msg )
    outfile.close()
