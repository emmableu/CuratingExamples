import urllib.request
import urllib.error
import json



import sys



def main():
    # print command line arguments
    # genre = sys.argv[1]
    # genres = ['animations', 'games']
    genres = ['games']
    project_ids = []
    page = 0
    for genre in genres:

        for i in range(0, 40):
            try:
                while page + 40 <= 200*(i+1):
                    url = "https://api.scratch.mit.edu/explore/projects?limit=40&offset=" \
                        + str(page) \
                        + "&language=en&mode=trending&q=" + genre
                    page += 40
                    req = urllib.request.Request(url, headers={'User-Agent': "Magic Browser"})
                    res = urllib.request.urlopen(req)
                    data = json.load(res)
                    print("Page: "+str(page))
                    for prj in data:
                        prj_id = prj['id']
                        project_ids.append(prj_id)
                        download_url = "https://projects.scratch.mit.edu/" + str(prj_id)
                        urllib.request.urlretrieve(download_url, genre + '/' + str(prj_id) + '.json')
            except urllib.error.HTTPError as e:
                print(e.fp.read())


if __name__ == "__main__":
    main()
