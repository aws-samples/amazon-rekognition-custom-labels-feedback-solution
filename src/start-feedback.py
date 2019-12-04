import boto3
from botocore.client import Config
from urllib.parse import urlparse
import os
import csv
import uuid
from botocore.exceptions import ClientError
import datetime
from decimal import Decimal
import json
from PIL import Image
from threading import Thread
import sys

runCommand = '--config feedback-config.json'

class AwsHelper:
    def getClient(self, name, awsRegion):
        config = Config(
            retries = dict(
                max_attempts = 5
            )
        )
        return boto3.client(name, region_name=awsRegion, config=config)
    
    def getResource(self, name, awsRegion=None):
        config = Config(
            retries = dict(
                max_attempts = 5
            )
        )

        if(awsRegion):
            return boto3.resource(name, region_name=awsRegion, config=config)
        else:
            return boto3.resource(name, config=config)

class S3Helper:

    @staticmethod
    def generatePresignedUrl(bucketName, fileName, awsRegion=None):  
        s3 = AwsHelper().getClient('s3', awsRegion)
        response = s3.generate_presigned_url('get_object',
                                            Params={'Bucket': bucketName,
                                            'Key': fileName},)
        return response

    
    @staticmethod
    def writeToS3(content, bucketName, s3FileName, awsRegion=None):
        s3 = AwsHelper().getResource('s3', awsRegion)
        object = s3.Object(bucketName, s3FileName)
        object.put(Body=content)

    @staticmethod
    def readFromS3(bucketName, s3FileName, awsRegion=None):
        s3 = AwsHelper().getResource('s3', awsRegion)
        obj = s3.Object(bucketName, s3FileName)
        return obj.get()['Body'].read().decode('utf-8')
    
    @staticmethod
    def readFromS3Uri(documentUri, awsRegion=None):
        o = urlparse(documentUri)
        bucketName = o.netloc
        fileName = o.path[1:]
        return S3Helper.readFromS3(bucketName, fileName)
    
    @staticmethod
    def parseBucketAndDocumentName(documentUri, awsRegion=None):
        o = urlparse(documentUri)
        bucketName = o.netloc
        fileName = o.path[1:]
        return (bucketName, fileName)
    
    @staticmethod
    def getImageSize(bucketName, imageName):
        s3 = AwsHelper().getResource('s3')
        bucket = s3.Bucket(bucketName)
        iobject = bucket.Object(imageName)
        iresponse = iobject.get()
        file_stream = iresponse['Body']
        im = Image.open(file_stream)
        return (im.width, im.height)
    
    @staticmethod
    def getS3BucketRegion(bucketName):
        client = boto3.client('s3')
        response = client.get_bucket_location(Bucket=bucketName)
        awsRegion = response['LocationConstraint']
        return awsRegion

    @staticmethod
    def getFileNames(awsRegion, bucketName, prefix, maxPages, allowedFileTypes):

        files = []

        currentPage = 1
        hasMoreContent = True
        continuationToken = None

        s3client = AwsHelper().getClient('s3', awsRegion)

        while(hasMoreContent and currentPage <= maxPages):
            if(continuationToken):
                listObjectsResponse = s3client.list_objects_v2(
                    Bucket=bucketName,
                    Prefix=prefix,
                    MaxKeys=1000,
                    ContinuationToken=continuationToken)
            else:
                listObjectsResponse = s3client.list_objects_v2(
                    Bucket=bucketName,
                    Prefix=prefix,
                    MaxKeys=1000)

            if(listObjectsResponse['IsTruncated']):
                continuationToken = listObjectsResponse['NextContinuationToken']
            else:
                hasMoreContent = False

            for doc in listObjectsResponse['Contents']:
                docName = doc['Key']
                docExt = FileHelper.getFileExtenstion(docName)
                docExtLower = docExt.lower()
                if(docExtLower in allowedFileTypes):
                    files.append(docName)
                    
            currentPage += 1

        return files
    
class FileHelper:
    @staticmethod
    def getFileNameAndExtension(filePath):
        basename = os.path.basename(filePath)
        dn, dext = os.path.splitext(basename)
        return (dn, dext[1:])

    @staticmethod
    def getFileName(fileName):
        basename = os.path.basename(fileName)
        dn, dext = os.path.splitext(basename)
        return dn

    @staticmethod
    def getFileExtenstion(fileName):
        basename = os.path.basename(fileName)
        dn, dext = os.path.splitext(basename)
        return dext[1:]

class ImageProcessor(Thread):

    def __init__(self, imageName, inputParameters, dataObject):
        ''' Constructor. '''
        Thread.__init__(self)
        self.imageName = imageName
        self.inputParameters = inputParameters
        self.dataObject = dataObject
        
    def getImageSize(self):
        s3 = AwsHelper().getResource('s3', self.inputParameters["awsRegion"])
        bucket = s3.Bucket(self.inputParameters["bucketName"])
        iobject = bucket.Object(self.imageName)
        iresponse = iobject.get()
        file_stream = iresponse['Body']
        im = Image.open(file_stream)
        return (im.width, im.height)

    def transformLabels(self, labels):
        fixedLabels = {}
        for ecl in labels["CustomLabels"]:
            lname = ecl["Name"]
            lconfidence = ecl["Confidence"]

            if(lname in fixedLabels):
                fixedLabel = fixedLabels[lname]
            else:
                fixedLabel = {}
                fixedLabel["Instances"] = []
                fixedLabels[lname] = fixedLabel
                fixedLabel["Name"] = lname

            if("Geometry" in ecl):
                fixedLabel["Instances"].append({"BoundingBox": ecl["Geometry"]["BoundingBox"], "Confidence": lconfidence})
            else:
                fixedLabel["Confidence"] = lconfidence

        clabels = []
        for efixedLabel in fixedLabels:
            fl = fixedLabels[efixedLabel]
            if(not "Confidence" in fl):
                fl["Confidence"] = -1
            clabels.append(fl)
        return clabels

    def run(self):
        try:
            print("Analyzing image: {}".format(self.imageName))
            imageWidth, imageHeight = self.getImageSize()
            self.dataObject["imageWidth"] = imageWidth
            self.dataObject["imageHeight"] = imageHeight

            rekognition = boto3.client('rekognition', region_name=self.inputParameters["awsRegion"])
            labels = rekognition.detect_custom_labels(
                Image={
                    'S3Object': {
                        'Bucket': self.inputParameters["bucketName"],
                        'Name': self.imageName,
                    }
                },
                ProjectVersionArn= self.inputParameters["projectVersionArn"],
                #MinConfidence = self.inputParameters["minimumConfidence"],
                #MaxResults=self.inputParameters["maxLabels"]
            )

            self.dataObject['labels'] = self.transformLabels(labels)
        except Exception as e:
            print("Failed to process labels for {}. Error: {}.".format(self.imageName, e))
            self.dataObject['labels'] = { 'Error' : "{}".format(e)}

class ImageAnalyzer:
    
    labelGroups = {}
    labelBoundingBoxGroups = {}
    noLabelsGroup = {}

    def __init__(self, images, inputParameters):
        ''' Constructor. '''
        self.images = images
        self.inputParameters = inputParameters
            
    def processLabel(self, imageName, imageUrl, imageWidth, imageHeight, label):
        instances = label['Instances']
        transformedInstances = []
        for einstance in instances:        
             transformedInstances.append((
                                round(einstance["BoundingBox"]["Left"]*imageWidth,2),
                                round(einstance["BoundingBox"]["Top"]*imageHeight,2),                            
                                round(einstance["BoundingBox"]["Width"]*imageWidth,2),
                                round(einstance["BoundingBox"]["Height"]*imageHeight,2))
            )

        imageMetadata = {}
        imageMetadata["imageLabelId"] = "{}-{}".format(imageUrl, label['Name'])
        imageMetadata["imageUrl"] = imageUrl
        imageMetadata["imageWidth"] = imageWidth
        imageMetadata["imageHeight"] = imageHeight
        imageMetadata["labelName"] = label['Name']
        imageMetadata["confidence"] = round(Decimal(label['Confidence']), 2)
        imageMetadata["instances"] = transformedInstances

        return imageMetadata
            
    def processLabels(self, dataObject):

        imageWidth = dataObject["imageWidth"]
        imageHeight = dataObject["imageHeight"]
        imageName = dataObject["imageName"]
        imageUrl = "s3://{}/{}".format(self.inputParameters["bucketName"], imageName)
        detectedLabels = dataObject["labels"]

        if(not detectedLabels):
            if(not imageUrl in self.noLabelsGroup):
                self.noLabelsGroup[imageUrl] = {"imageUrl": imageUrl, "imageWidth": imageWidth, "imageHeight": imageHeight}
        else:
            for label in detectedLabels:
                metadata = self.processLabel(imageName, imageUrl, imageWidth, imageHeight, label)

                if(label['Instances']):            
                    if(label["Name"] in self.labelBoundingBoxGroups):
                        ig = self.labelBoundingBoxGroups[label["Name"]]
                        ig.append(metadata)
                    else:
                        ig = []
                        ig.append(metadata)
                        self.labelBoundingBoxGroups[label["Name"]] = ig
                else:            
                    if(label["Name"] in self.labelGroups):
                        lg = self.labelGroups[label["Name"]]
                        lg.append(metadata)
                    else:
                        lg = []
                        lg.append(metadata)
                        self.labelGroups[label["Name"]] = lg
  
    def processBatch(self, threads):
        for thr in threads:
            thr.start()

        for thr in threads:
            thr.join()

    def run(self):
        
        threads = []
        output = []
        
        totalImages = len(self.images)
        
        i = 1
        for imageName in self.images:
            
            ado = { 'imageName' : imageName }
            
            output.append(ado)
            ip = ImageProcessor(imageName, self.inputParameters, ado)
            threads.append(ip)
            
            if(i % self.inputParameters["concurrencyControl"] == 0):
                self.processBatch(threads)
                for dataObject in output:
                    self.processLabels(dataObject)
                print("Analyzed images: {}/{}".format(i, totalImages))
                output.clear()
                threads.clear()

            i = i + 1
            
        if(threads):
            self.processBatch(threads)
            for dataObject in output:
                    self.processLabels(dataObject)
            print("Analyzed images: {}/{}".format(i-1, totalImages))
            output.clear()
            threads.clear()
        
        return (self.labelGroups, self.labelBoundingBoxGroups, self.noLabelsGroup)
        
class BoundingBoxScheduler:

    def __init__(self, labelBoundingBoxGroups, inputParameters):
        self.labelBoundingBoxGroups = labelBoundingBoxGroups
        self.inputParameters = inputParameters
        
    def getHtmlTemplate(self, headerText=None, fullInstructions=None, shortInstructions=None):

        if(not headerText):
            headerText = "Please adjust existing bounding box around instances of humans and assign the correct label. See full instructions for additional information."

        if(not fullInstructions):
            fullInstructions = """
        <full-instructions header="Bounding box adjustment instructions">
            <p>Note: For this task, if there are more than 4 people in the image, you only need to label the closest/biggest 4 people. </p><p><br></p><ol><li><strong>Inspect</strong> the image</li><li><strong>Determine</strong> if the specified label is/are visible in the picture.</li><li><strong>Outline</strong> each instance of the specified label in the image using the provided “Box” tool.</li></ol><ul><li>Boxes should fit tight around each object</li><li>Do not include parts of the object are overlapping or that cannot be seen, even though you think you can interpolate the whole shape.</li><li>Avoid including shadows.</li><li>If the target is off screen, draw the box up to the edge of the image.</li></ul><p><img src="https://d1i6hezpxab4vs.cloudfront.net/76082909-991b-406e-83a3-92726c2b00c5/src/images/bounding-box-good-example.png" style="max-width:100%"></p><h2><span style="color: rgb(0, 138, 0);">Good Example</span></h2><p><img src="https://d1i6hezpxab4vs.cloudfront.net/76082909-991b-406e-83a3-92726c2b00c5/src/images/bounding-box-bad-example.png" style="max-width:100%"></p><h2><span style="color: rgb(230, 0, 0);">Bad Example</span></h2>
        </full-instructions>"""
        if(not shortInstructions):
            shortInstructions = """
        <short-instructions>
            <h2><span style="color: rgb(0, 138, 0);">Good example</span></h2><p>Enter description of a correct bounding box label</p><p><img src="https://d1i6hezpxab4vs.cloudfront.net/76082909-991b-406e-83a3-92726c2b00c5/src/images/quick-instructions-example-placeholder.png" style="max-width:100%"></p><p><br></p><h2><span style="color: rgb(230, 0, 0);">Bad example</span></h2><p>Enter description of an incorrect bounding box label</p><p><img src="https://d1i6hezpxab4vs.cloudfront.net/76082909-991b-406e-83a3-92726c2b00c5/src/images/quick-instructions-example-placeholder.png" style="max-width:100%"></p>
        </short-instructions>"""    

        htmlTemplate = """
    <script src="https://assets.crowd.aws/crowd-html-elements.js"></script>
    <crowd-form>
      <crowd-bounding-box
        name="boundingBox"
        src="{{{{ task.input.taskObject | grant_read_access }}}}"
        header="{}"
        labels="{{{{ task.input.labels | to_json | escape }}}}"
        initial-value="[
          {{% for box in task.input.manifestLine.bounding-box.annotations %}}
            {{% capture class_id %}}{{{{ box.class_id }}}}{{% endcapture %}}
            {{% assign label = task.input.manifestLine.bounding-box-metadata.class-map[class_id] %}}
          {{
            label: {{{{label | to_json}}}},
            left: {{{{box.left}}}},
            top: {{{{box.top}}}},
            width: {{{{box.width}}}},
            height: {{{{box.height}}}},
          }},
          {{% endfor %}}
        ]"
      >
        {}{}
      </crowd-bounding-box>
    </crowd-form>""".format(headerText, fullInstructions, shortInstructions)

        return htmlTemplate

    def createManifestGroups(self):
        
        manifestGroups = []

        manifestGroup = None

        maxLabels = self.inputParameters["maxLabelsPerBoundingBoxJob"]

        labelId = 0
        for label in self.labelBoundingBoxGroups:

            if(labelId % maxLabels == 0):
                manifestGroup = {}
                manifestGroup["items"] = {}
                manifestGroup["labels"] = {}
                manifestGroups.append(manifestGroup)

            manifestItems = manifestGroup["items"]
            manifestLabels = manifestGroup["labels"]

            if(not labelId in manifestLabels):
                manifestLabels[labelId] = label

            for eimage in self.labelBoundingBoxGroups[label]:
                if(eimage["imageUrl"] in manifestItems):
                    manifestItem = manifestItems[eimage["imageUrl"]]
                else:
                    manifestItem = {}
                    manifestItem["imageUrl"] = eimage["imageUrl"]
                    manifestItem["imageWidth"] = eimage["imageWidth"]
                    manifestItem["imageHeight"] = eimage["imageHeight"]
                    manifestItem["annotations"] = []
                    manifestItem["confidences"] = []
                    manifestItem["classMap"] = {}
                    manifestItems[eimage["imageUrl"]] = manifestItem

                for einstance in eimage["instances"]:
                    manifestItem["annotations"].append({"class_id": labelId,
                                        "left": int(einstance[0]),
                                        "top": int(einstance[1]),
                                        "width": int(einstance[2]),
                                        "height": int(einstance[3])})

                    manifestItem["confidences"].append(0.9)

                if(not labelId in manifestItem["classMap"]):
                    manifestItem["classMap"][labelId] = label

            labelId += 1

        return manifestGroups
            
    def createManifestFiles(self, manifestGroups):
        
        i = 0

        manifestFiles = []

        for manifestGroup in manifestGroups:

            manifestItemsText = ""

            manifestFileName = "{}/manifest-{}.json".format(self.inputParameters["boundingBoxManifestPath"], i)
            labelsFileName = "{}/labels-{}.json".format(self.inputParameters["boundingBoxManifestPath"], i)    
            htmlFileName = "{}/html-template-{}.html".format(self.inputParameters["boundingBoxManifestPath"], i)

            groupLabels = []
            for label in manifestGroup["labels"]:
                groupLabels.append({"label": manifestGroup["labels"][label]})

            labels = {
                "document-version": "2018-11-28", 
                "labels": groupLabels
            }

            S3Helper.writeToS3(json.dumps(labels), self.inputParameters["outputBucket"], labelsFileName)

            S3Helper.writeToS3(self.getHtmlTemplate(), self.inputParameters["outputBucket"], htmlFileName)

            for manifestItemUrl in manifestGroup["items"]:
                manifestItem = manifestGroup["items"][manifestItemUrl]

                annotations = []
                for eannotation in manifestItem["annotations"]:
                    annotations.append({
                        "class_id": eannotation["class_id"],
                        "left": eannotation["left"],
                        "top": eannotation["top"],
                        "width": eannotation["width"],
                        "height": eannotation["height"]})

                confidences = []
                for econfidence in manifestItem["confidences"]:
                    confidences.append({"confidence": econfidence})

                classMap = {}
                for eclass in manifestItem["classMap"]:
                    classMap[eclass] = manifestItem["classMap"][eclass]

                manifestItemJSON = { 
                    "source-ref": manifestItem["imageUrl"],
                    "bounding-box": {
                        "image_size": [{ "width":  manifestItem["imageWidth"], "height": manifestItem["imageHeight"], "depth": 3 }],
                        "annotations": annotations
                    },
                    "bounding-box-metadata": {
                        "objects": confidences,
                        "class-map": classMap,
                        "type": "groundtruth/object-detection",
                        "job-name": "labeling-job/test"
                    }
                }

                manifestItemsText += json.dumps(manifestItemJSON) + "\n"

            S3Helper.writeToS3(manifestItemsText, self.inputParameters["outputBucket"], manifestFileName)

            manifestFiles.append({"manifest": "s3://{}/{}".format(self.inputParameters["outputBucket"], manifestFileName),
                                  "labels-manifest": "s3://{}/{}".format(self.inputParameters["outputBucket"], labelsFileName),
                                 "html-template": "s3://{}/{}".format(self.inputParameters["outputBucket"], htmlFileName)})

            i += 1
        return manifestFiles
    
    def createGTJob(self, jobName, manifestUri, labelsUri, outputUri, preLambda, postLambda, roleArn, workTeamArn, templateUri):

        sageMakerClient = AwsHelper().getClient("sagemaker", self.inputParameters["awsRegion"])

        response = sageMakerClient.create_labeling_job(
            LabelingJobName=jobName,
            LabelAttributeName="bounding-box-new",
            InputConfig={
                'DataSource': {
                    'S3DataSource': {
                        'ManifestS3Uri': manifestUri
                    }
                },
                'DataAttributes': {
                    'ContentClassifiers': [
                        'FreeOfPersonallyIdentifiableInformation'
                    ]
                }
            },
            OutputConfig={
                'S3OutputPath': outputUri
            },
            RoleArn=roleArn,
            LabelCategoryConfigS3Uri=labelsUri,
            HumanTaskConfig={
                'WorkteamArn': workTeamArn,
                 'UiConfig': {
                    'UiTemplateS3Uri': templateUri
                 },
                'PreHumanTaskLambdaArn': preLambda,
                'TaskTitle': 'Confirm Bounding Boxes',
                'TaskDescription': 'Confirm bounding boxes.',
                'NumberOfHumanWorkersPerDataObject': 1,
                'TaskTimeLimitInSeconds': 600,
                'MaxConcurrentTaskCount': 10,
                'AnnotationConsolidationConfig': {
                    'AnnotationConsolidationLambdaArn': postLambda
                }
            }
        )

    def createBoundingBoxGTJobs(self, manifestFiles):

        jobs = []
        mindex = 0

        lambdaMap = {
            "us-east-1": "432418664414",
            "us-east-2": "266458841044",
            "us-west-2": "081040173940",
            "ca-central-1": "918755190332",
            "eu-west-1": "568282634449",
            "eu-west-2": "487402164563",
            "eu-central-1": "203001061592",
            "ap-northeast-1": "477331159723",
            "ap-northeast-2": "845288260483",
            "ap-south-1": "565803892007",
            "ap-southeast-1": "377565633583",
            "ap-southeast-2": "454466003867"
        }

        for manifestFile in manifestFiles:

            #jobName = "{}".format(uuid.uuid1())
            jobName = "{}-{}".format(self.inputParameters["runId"], mindex)
            manifestUri = manifestFile["manifest"]
            labelsUri = manifestFile["labels-manifest"]
            outputUri = "s3://{}/{}".format(self.inputParameters["outputBucket"], self.inputParameters["boundingBoxGroundTruthOutputPath"])
            templateUri = manifestFile["html-template"]

            roleArn = self.inputParameters["gtJobRoleArn"]
            workTeamArn = self.inputParameters["gtWorkTeamArn"]
            
            preLambda = "arn:aws:lambda:{}:{}:function:PRE-AdjustmentBoundingBox".format(self.inputParameters["awsRegion"], lambdaMap[self.inputParameters["awsRegion"]])
            postLambda = "arn:aws:lambda:{}:{}:function:ACS-AdjustmentBoundingBox".format(self.inputParameters["awsRegion"], lambdaMap[self.inputParameters["awsRegion"]])

            # preLambda = "arn:aws:lambda:us-east-1:432418664414:function:PRE-AdjustmentBoundingBox"
            # postLambda = "arn:aws:lambda:us-east-1:432418664414:function:ACS-AdjustmentBoundingBox"

            self.createGTJob(jobName, manifestUri, labelsUri, outputUri, preLambda, postLambda, roleArn, workTeamArn, templateUri)

            jobs.append({ "jobName": jobName, "manifestUri" : manifestUri, "labelsUri": labelsUri,
                        "templateUri": templateUri, "outputUri": outputUri})

            mindex += 1
            
        return jobs

    def run(self):
        print("Label BoundingBox Groups:")
        for ebbLabel in self.labelBoundingBoxGroups:
            print("BBLabel: {}".format(ebbLabel))
            for eimage in self.labelBoundingBoxGroups[ebbLabel]:
                print(eimage["imageUrl"])

        manifestGroups = self.createManifestGroups()
        manifestFiles = self.createManifestFiles(manifestGroups)
        jobs = self.createBoundingBoxGTJobs(manifestFiles)
        return jobs

class LabelVerificationScheduler:

    def __init__(self, labelGroups, inputParameters):
        self.labelGroups = labelGroups
        self.inputParameters = inputParameters

    def createManifestFiles(self):
        imageBatchSize = self.inputParameters["maxImagesPerLabelVerificationBatch"]
        masterManifestFileText = ""
        for elabel in self.labelGroups:
            i = 0
            j = 0
            # print("Generating manifest files for label: {}".format(elabel))
            
            manifestItems = []
            
            for eimage in self.labelGroups[elabel]:
                manifestItems.append({"imageUrl": eimage["imageUrl"], 
                                    "label": elabel,
                                    "confidence" : float(eimage["confidence"])})
                
                i += 1
                
                if( i % imageBatchSize == 0):
                    fileName = "{}/manifest-{}-{}.json".format(self.inputParameters["labelManifestPath"], elabel.replace(" ", "-"), j)
                    S3Helper.writeToS3(json.dumps(manifestItems), self.inputParameters["outputBucket"], fileName)
                    # print("s3://{}/{}".format(self.inputParameters["outputBucket"], fileName))
                    masterManifestFileText += ('{{ "source-ref": "s3://{}/{}" }}\n'.format(self.inputParameters["outputBucket"], fileName))
                    manifestItems.clear()
                    j += 1
                    
            if(manifestItems):
                fileName = "{}/manifest-{}-{}.json".format(self.inputParameters["labelManifestPath"], elabel.replace(" ", "-"), j)
                S3Helper.writeToS3(json.dumps(manifestItems), self.inputParameters["outputBucket"], fileName)
                # print("s3://{}/{}".format(self.inputParameters["outputBucket"], fileName))
                masterManifestFileText += ('{{ "source-ref": "s3://{}/{}" }}\n'.format(self.inputParameters["outputBucket"], fileName))
                manifestItems.clear()
                j += 1
                
        labelVerificationManifestFileName = "{}/manifest.json".format(self.inputParameters["labelManifestPath"])

        S3Helper.writeToS3(masterManifestFileText, self.inputParameters["outputBucket"], labelVerificationManifestFileName)

        print("Generated label verification manifest file...")
        print("s3://{}/{}".format(self.inputParameters["outputBucket"], labelVerificationManifestFileName))

        return labelVerificationManifestFileName

    def getLabelVeriificationHtmlTemplate(self):
        
        labelVerificationHtmlTemplate = """
    <script src="https://assets.crowd.aws/crowd-html-elements.js"></script>
    <style>
    .center {    text-align: center;
    border: 3px solid white;
    }
    .row {    display: flex;
    flex-wrap: wrap;
    padding: 0 4px;
    }
    /* Create five equal columns that sits next to each other */
    .column {    flex: 18%;
    padding: 0 4px;
    }
    </style>
    <crowd-form>
    <div class="center">
    <h1> Confirm that each image is correctly labelled as "{{ task.input.sourceRef[0].label }}"</h1>
    </div>
    <div class="row">
    {% assign length = task.input.sourceRef.size | minus: 1 %}
    {% for i in (0..length) %}
    <div class="column">
    <crowd-card
    image="{{ task.input.sourceRef[i].imageUrl | grant_read_access }}">
    <div class="center">Confirm <crowd-checkbox checked="true" name="item-{{i}}" value="Confirmed" /></div>
    </crowd-card>
    </div>
    {% endfor %}
    </div>
    </crowd-form>"""

        return labelVerificationHtmlTemplate

    def createCutomGTJob(self, jobName, manifestUri, outputUri, preLambda, postLambda, roleArn, workTeamArn, templateUri):
        
        sageMakerClient = AwsHelper().getClient("sagemaker", self.inputParameters["awsRegion"])

        response = sageMakerClient.create_labeling_job(
            LabelingJobName=jobName,
            LabelAttributeName="labels",
            InputConfig={
                'DataSource': {
                    'S3DataSource': {
                        'ManifestS3Uri': manifestUri
                    }
                },
                'DataAttributes': {
                    'ContentClassifiers': [
                        'FreeOfPersonallyIdentifiableInformation'
                    ]
                }
            },
            OutputConfig={
                'S3OutputPath': outputUri
            },
            RoleArn=roleArn,
            HumanTaskConfig={
                'WorkteamArn': workTeamArn,
                'UiConfig': {
                    'UiTemplateS3Uri': templateUri
                },
                'PreHumanTaskLambdaArn': preLambda,
                'TaskTitle': 'Confirm label for images below',
                'TaskDescription': 'Confirm images for label.',
                'NumberOfHumanWorkersPerDataObject': 1,
                'TaskTimeLimitInSeconds': 600,
                'MaxConcurrentTaskCount': 10,
                'AnnotationConsolidationConfig': {
                    'AnnotationConsolidationLambdaArn': postLambda
                }
            }
        )

    def run(self):
        
        labelVerificationManifestFileName = self.createManifestFiles()

        labelVerificationHtmlTemplate = self.getLabelVeriificationHtmlTemplate()
        labelVerificationHtmlTemplateFile = "{}/html-template.html".format(self.inputParameters["labelManifestPath"])
        S3Helper.writeToS3(labelVerificationHtmlTemplate, self.inputParameters["outputBucket"], labelVerificationHtmlTemplateFile)

        jobName = "{}".format(uuid.uuid1())

        manifestUri = "s3://{}/{}".format(self.inputParameters["outputBucket"], labelVerificationManifestFileName)
        outputUri = "s3://{}/{}".format(self.inputParameters["outputBucket"], self.inputParameters["labelGroundTruthOutputPath"])
        templateUri = "s3://{}/{}".format(self.inputParameters["outputBucket"], labelVerificationHtmlTemplateFile)

        roleArn = self.inputParameters["gtJobRoleArn"]
        workTeamArn = self.inputParameters["gtWorkTeamArn"]

        preLambda = self.inputParameters["gtLabelVerificationPreLambda"]
        postLambda = self.inputParameters["gtLabelVerificationPostLambda"]

        self.createCutomGTJob(jobName, manifestUri, outputUri, preLambda, postLambda, roleArn, workTeamArn, templateUri)

        return jobName

class JobScheduler:

    def __init__(self, inputParameters):
        self.inputParameters = inputParameters
   
    def printGroups(self, labelGroups, labelBoundingBoxGroups, noLabelsGroup):
        print("No Labels")
        print(noLabelsGroup)
        
        print("Label Groups:")
        for elabel in labelGroups:
            print("Label: {}".format(elabel))
            for eimage in labelGroups[elabel]:
                print(eimage["imageUrl"])
                
        print("Label BoundingBox Groups:")
        for ebbLabel in labelBoundingBoxGroups:
            print("BBLabel: {}".format(ebbLabel))
            for eimage in labelBoundingBoxGroups[ebbLabel]:
                print(eimage["imageUrl"])

    def setOutputPaths(self, runId):
        self.inputParameters["labelManifestPath"] = "datasets/{}/label-verification/manifest".format(runId)
        self.inputParameters["labelGroundTruthOutputPath"] = "datasets/{}/label-verification/ground-truth-output".format(runId)
        self.inputParameters["boundingBoxManifestPath"] = "datasets/{}/bounding-box-verification/manifest".format(runId)
        self.inputParameters["boundingBoxGroundTruthOutputPath"] = "datasets/{}/bounding-box-verification/ground-truth-output".format(runId)
        self.inputParameters["noLabelsManifestPath"] = "datasets/{}/no-labels/manifest".format(runId)
        self.inputParameters["jobsListPath"] = "datasets/{}/jobs".format(runId)

    def parseInputPath(self):
        print("Input data path: {}".format(self.inputParameters["datasetPath"]))
        bucketName, inputDocumentPath = S3Helper.parseBucketAndDocumentName(self.inputParameters["datasetPath"])
        self.inputParameters["bucketName"] = bucketName
        self.inputParameters["inputDocumentPath"] = inputDocumentPath
        print("Images bucket: {}, Images path: {}".format(bucketName, inputDocumentPath))

    def getImageList(self):
        print("Getting image list...")
        allowedFileTypes = ["jpg", "jpeg", "png"]
        images = S3Helper.getFileNames(self.inputParameters["awsRegion"], self.inputParameters["bucketName"],
                                     self.inputParameters["inputDocumentPath"], 100, allowedFileTypes)
        print("Total images: {}".format(len(images)))
        return images

    def startBoundingBoxAdjustmentJobs(self, labelBoundingBoxGroups):
        print("Starting bounding box adjustment jobs...")
        boundingBoxJobScheduler = BoundingBoxScheduler(labelBoundingBoxGroups, self.inputParameters)
        boundingBoxJobs = boundingBoxJobScheduler.run()
        print("Started {} jobs for bounding box adjustment.".format(len(boundingBoxJobs)))
        client = AwsHelper().getClient('sagemaker', self.inputParameters["awsRegion"])
        for job in boundingBoxJobs:
            response = client.describe_labeling_job(LabelingJobName=job["jobName"])
            print("Job: {}, Status: {}".format(job["jobName"], response["LabelingJobStatus"]))
        return boundingBoxJobs

    def startLabelVerificationJobs(self, labelGroups):
        print("Starting label verification job.")
        labelVerificationScheduler = LabelVerificationScheduler(labelGroups, self.inputParameters)
        labelVerificationJob = labelVerificationScheduler.run()
        print("Started label verification job with Id: {}".format(labelVerificationJob))
        client = AwsHelper().getClient('sagemaker', self.inputParameters["awsRegion"])
        response = client.describe_labeling_job(LabelingJobName=labelVerificationJob)
        print("Job: {}, Status: {}".format(labelVerificationJob, response["LabelingJobStatus"]))
        return labelVerificationJob

    def generateOutputJobsFile(self, boundingBoxJobs, labelVerificationJob, noLabelsFile):
        jobsList = {}

        bbvjobs = []
        for job in boundingBoxJobs:
            bbvjobs.append(job["jobName"])
        jobsList["runid"] = self.inputParameters["runId"]
        jobsList["bounding-box-verification-jobs"] = bbvjobs
        jobsList["label-verification-job"] = labelVerificationJob
        jobsList["no-labels-manifest-file"] = noLabelsFile
        # print(jobsList)
        
        jobsListFile = "{}/jobs.json".format(self.inputParameters["jobsListPath"])
        S3Helper.writeToS3(json.dumps(jobsList), self.inputParameters["outputBucket"], jobsListFile)
        jobsListFile = "{}/jobs.json".format(self.inputParameters["jobsListPath"])
        S3Helper.writeToS3(json.dumps(jobsList), self.inputParameters["outputBucket"], jobsListFile)
        
        print("Jobs manifest generated: s3://{}/{}".format(self.inputParameters["outputBucket"], jobsListFile))

        print("=============================================")
        print("To genarete final output with human review run command below:\npython3 get-feedback.py --jobs-manifest ""s3://{}/{}""".format(self.inputParameters["outputBucket"], jobsListFile))
        print("=============================================")

    def createNoLabelsManifest(self, noLabelsGroup):

        s3FilePath = ""

        if(noLabelsGroup):
            fileText = ""
            for elabelKey in noLabelsGroup:
                elabel = noLabelsGroup[elabelKey]
                item = {}
                item["source-ref"] = elabel["imageUrl"]
                item["nolabel"] = {"annotations": [], "image_size": [{"width":elabel["imageWidth"],"depth":3,"height":elabel["imageHeight"]}]}
                item["nolabel-metadata"] = {"job-name":"labeling-job/nolabels",
                                                "class-map":{},
                                                "human-annotated":"yes",
                                                "objects":[],
                                                "creation-date":"2019-11-27T10:49:14.678944"
                                                ,"type":"groundtruth/object-detection"}
                fileText += json.dumps(item) + "\n"
            
            noLabelsManifestFile = "{}/nolabels.json".format(self.inputParameters["noLabelsManifestPath"])
            S3Helper.writeToS3(fileText, self.inputParameters["outputBucket"], noLabelsManifestFile)
            s3FilePath = "s3://{}/{}".format(self.inputParameters["outputBucket"], noLabelsManifestFile)

        return s3FilePath

    def run(self):

        #Run Id
        runId = str(uuid.uuid1())
        self.inputParameters["runId"] = runId
        print("Run Id: {}".format(runId))
        print("AWS Region: {}".format(self.inputParameters["awsRegion"]))
        self.setOutputPaths(runId)
        self.parseInputPath()
        images = self.getImageList()
        
        # Analyze images
        print("Analyzing images...")
        imageAnalyzer = ImageAnalyzer(images, self.inputParameters)
        labelGroups, labelBoundingBoxGroups, noLabelsGroup = imageAnalyzer.run()
        # self.printGroups(labelGroups, labelBoundingBoxGroups, noLabelsGroup)
        
        noLabelsFile = ""
        if(noLabelsGroup):
            noLabelsFile = self.createNoLabelsManifest(noLabelsGroup)

        # Start GT jobs
        boundingBoxJobs = []
        if(labelBoundingBoxGroups):
            boundingBoxJobs = self.startBoundingBoxAdjustmentJobs(labelBoundingBoxGroups)
        labelVerificationJob = ""
        if(labelGroups):
            labelVerificationJob = self.startLabelVerificationJobs(labelGroups)
        
        #Output job file
        self.generateOutputJobsFile(boundingBoxJobs, labelVerificationJob, noLabelsFile)

class CustomLabelsFeedback:
    
    def __init__(self):
        ''' Constructor. '''
        print("")

    def validateInput(self, input):

        event = {}

        #Validate input parameters
        event['datasetPath'] = input["images"]
        event["outputBucket"] = input["outputBucket"]
        event["gtJobRoleArn"] = input["jobRoleArn"]
        event["gtWorkTeamArn"] = input["workforceTeamArn"]
        event["gtLabelVerificationPreLambda"] = input["preLambdaArn"]
        event["gtLabelVerificationPostLambda"] = input["postLambdaArn"]
        event["projectVersionArn"] = input["projectVersionArn"]
        event["concurrencyControl"] = input["concurrencyControl"]
        event["minimumConfidence"] = input["minimumConfidence"]
        event["maxLabelsPerBoundingBoxJob"] = input["maxLabelsPerBoundingBoxJob"]
        event["maxImagesPerLabelVerificationBatch"] = input["maxImagesPerLabelVerificationBatch"]
        event["maxLabels"] = input["maxLabels"]

        awsRegion = 'us-east-1'
        bucketName, documentPath = S3Helper.parseBucketAndDocumentName(event['datasetPath'])
        ar = S3Helper.getS3BucketRegion(bucketName)
        if(ar):
            awsRegion = ar

        event['awsRegion'] = awsRegion
        event['bucketName'] = bucketName
        event['documentPath'] = documentPath

        return event

    def getInput(self, args):

        inputFile = "feedback-config.json"

        i = 0
        while(i < len(args)):
            if(args[i] == '--config'):
                inputFile = args[i+1]
                i = i + 1
            i += 1
        
        with open(inputFile, 'r') as i:
            input = json.load(i)

        return input

    def run(self, args):
        event = self.validateInput(self.getInput(args))
        jobScheduler = JobScheduler(event)
        jobScheduler.run()

try:
    cliMode = True

    if cliMode:
        args = sys.argv
    else:
        args = runCommand.split(' ')

    clf = CustomLabelsFeedback()
    clf.run(args)
except Exception as e:
    print("Something went wrong:\n====================================================\n{}".format(e))