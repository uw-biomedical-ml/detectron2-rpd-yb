import tornado.ioloop
import tornado.web
import json, glob, random, os
import pandas as pd

allocts = {}
allirs = {}
regradeData = None

def loadData():
    global allocts,allirs
    df = pd.read_parquet('/data/amd-data/cera-rpd/detectron2-rpd/datasets/dfUWAMD_dfsemisup_preinj_sample_toserve.parquet')
    df.loc[:,'img_path'] = ['/images' + s for s in df.img_path]
    df.loc[:,'ir_path'] = ['/images' + s for s in df.ir_path]
    allocts = df[['instance','img_path']].set_index('img_path').groupby('instance').groups
    allirs = df[['instance','ir_path']].drop_duplicates().set_index('ir_path').groupby('instance').groups
    print(len(allocts.keys()))



def checkfinished(user):
    if not os.path.isfile(f"{user}.tsv"):
        with open(f"{user}.tsv", "w") as fout:
            fout.write("id\tgrades\n")
        return set()
    done = set()
    with open(f"{user}.tsv") as fin:
        header = None
        for l in fin:
            if header == None:
                header = l
                continue
            done.add(l.split("\t")[0])
    return done


def getNext(user):
    done = checkfinished(user)
    remaining = allocts.keys() - done
    return random.choice(tuple(remaining)), len(remaining), len(allocts.keys())

def record(user, data, volid):
    with open(f"{user}.tsv", "a") as fout:
        fout.write(f"{volid}\t{json.dumps(data)}\n")



class Login(tornado.web.RequestHandler):
    def get(self):
        self.write('<html><body>User: <form action="/grade" method="POST">'
                   '<input type="text" name="user">'
                   '<input type="hidden" name="method" value="start">'
                   '<input type="submit" value="Submit">'
                   '</form></body></html>')

class Grading(tornado.web.RequestHandler):
    def post(self):
        method = self.get_body_argument("method")
        user = self.get_body_argument("user")
        if method == "record":
            volid = self.get_body_argument("volid")
            data = {}
            found = 0
            for f in ("ft_RPD", "ft_noRPD", "ft_Unable"):
                try:
                    data[f] = self.get_body_argument(f)
                    found += 1
                except:
                    data[f] = "off"
                    pass
            if found != 0:
                record(user, data, volid)
        inext, remaining, total = getNext(user)
        images = allocts[inext]
        irs = allirs[inext]
        imghtml = ""
        for fn in irs:
            imghtml += f"<img src='{fn}' width='1000' />"
        for fn in images:
            imghtml += f"<img src='{fn}' width='430' />"

        with open('index.html') as fin:
            indexstr = fin.read()
        indexstr = indexstr.replace("######", imghtml)
        indexstr = indexstr.replace("<user>", user)
        indexstr = indexstr.replace("<remaining>", str(remaining))
        indexstr = indexstr.replace("<total>", str(total))
        indexstr = indexstr.replace("<action>", "/grade")
        indexstr = indexstr.replace("<volid>", inext)
        self.write(indexstr)

def make_app():
    return tornado.web.Application([
        (r"/", Login),
        (r"/grade", Grading),
        (r"/images/(.*)", tornado.web.StaticFileHandler, {'path': "/"})
    ])

if __name__ == "__main__":
    loadData()
    app = make_app()
    app.listen(8766)
    tornado.ioloop.IOLoop.current().start()

