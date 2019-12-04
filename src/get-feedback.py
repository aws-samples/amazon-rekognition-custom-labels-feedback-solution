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
import sys
import time

runCommand = 'python3 process-jobs.py --jobs-manifest '

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
                    MaxKeys=10,
                    ContinuationToken=continuationToken)
            else:
                listObjectsResponse = s3client.list_objects_v2(
                    Bucket=bucketName,
                    Prefix=prefix,
                    MaxKeys=10)

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

class BoundingBoxVerificationJobProcessor:

    def __init__(self, inputParameters):
        self.inputParameters = inputParameters

    def generateOutput(self, automlManifestItems):

        automlText = ""
        for automlManifestItemKey in automlManifestItems:
            automlManifestItem = automlManifestItems[automlManifestItemKey]
            
            automlText += json.dumps(automlManifestItem) + "\n"

        S3Helper.writeToS3(automlText, self.inputParameters["outputBucket"], self.inputParameters["boundingBoxOutputFile"])

        # print(S3Helper.generatePresignedUrl(self.inputParameters["outputBucket"], self.inputParameters["boundingBoxOutputFile"]))

    def processJobResults(self, boundingBoxJobs):
        automlManifestItems = {}

        sageMakerClient = AwsHelper().getClient("sagemaker", self.inputParameters["awsRegion"])

        for job in boundingBoxJobs:
            response = sageMakerClient.describe_labeling_job(LabelingJobName=job)        
            outputManifestUri = response["LabelingJobOutput"]["OutputDatasetS3Uri"]
            outputBucketName, outputFileName = S3Helper.parseBucketAndDocumentName(outputManifestUri)
            jobOutputText = S3Helper.readFromS3(outputBucketName, outputFileName)        
            outputManifestItems = jobOutputText.splitlines()
            
            for outputManifestItemText in outputManifestItems:
                eoutputManifestItem = json.loads(outputManifestItemText)
                
                if(eoutputManifestItem["source-ref"] in automlManifestItems):
                    automlManifestItem = automlManifestItems[eoutputManifestItem["source-ref"]]
                    oid = uuid.uuid1()
                    automlManifestItem["{}-bounding-box-new".format(oid)] = eoutputManifestItem["bounding-box-new"]
                    automlManifestItem["{}-bounding-box-new-metadata".format(oid)] = eoutputManifestItem["bounding-box-new-metadata"]
                else:
                    automlManifestItem =  { "source-ref" : eoutputManifestItem["source-ref"], 
                                            "bounding-box-new": eoutputManifestItem["bounding-box-new"],
                                            "bounding-box-new-metadata": eoutputManifestItem["bounding-box-new-metadata"]
                                            }
                    automlManifestItems[eoutputManifestItem["source-ref"]] = automlManifestItem

        self.generateOutput(automlManifestItems)

    def checkJobStatus(self, boundingBoxJobs):
        sageMakerClient = AwsHelper().getClient("sagemaker", self.inputParameters["awsRegion"])
    
        for job in boundingBoxJobs:
            response = sageMakerClient.describe_labeling_job(LabelingJobName=job)
            print("Job: {}, Status: {}".format(job, response["LabelingJobStatus"]))
            jobStatus = response["LabelingJobStatus"]
            while (jobStatus == "InProgress"):
                time.sleep(10)
                response = sageMakerClient.describe_labeling_job(LabelingJobName=job)
                print("Job: {}, Status: {}".format(response["LabelingJobName"], response["LabelingJobStatus"]))
                jobStatus = response["LabelingJobStatus"]

    def run(self, boundingBoxJobs):
        self.checkJobStatus(boundingBoxJobs)
        self.processJobResults(boundingBoxJobs)

class LabelVerificationJobProcessor:

    def __init__(self, inputParameters):
        self.inputParameters = inputParameters

    def generateOutput(self, finalManifestItems, labelVerificationJobName):
        finalJobOutput = ""

        for finalManifestItemKey in finalManifestItems:
            finalLabels = finalManifestItems[finalManifestItemKey]
            
            label = {
                "source-ref": finalManifestItemKey
            }

            i = 0
            for finalLabel in finalLabels:
                
                label["label-{}".format(i)] = "0"
                label["label-{}-metadata".format(i)] = {
                    "class-name": "{}".format(finalLabel["label"]),
                    "confidence": finalLabel["confidence"],
                    "type":"groundtruth/image-classification",
                    "job-name": labelVerificationJobName,
                    "human-annotated": "yes",
                    "creation-date": "2018-10-18T22:18:13.527256"
                }
                
                i += 1
            finalJobOutput += json.dumps(label) + "\n"

        S3Helper.writeToS3(finalJobOutput, self.inputParameters["outputBucket"], self.inputParameters["labelsOutputFile"])

        #print(S3Helper.generatePresignedUrl(self.inputParameters["outputBucket"], self.inputParameters["labelsOutputFile"]))

    def processJobResults(self, labelVerificationJobName):
        
        finalManifestItems = {}

        sageMakerClient = AwsHelper().getClient("sagemaker", self.inputParameters["awsRegion"])
        response = sageMakerClient.describe_labeling_job(LabelingJobName=labelVerificationJobName)

        outputManifestUri = response["LabelingJobOutput"]["OutputDatasetS3Uri"]

        jobOutputText = S3Helper.readFromS3Uri(outputManifestUri)

        outputManifestItems = jobOutputText.splitlines()

        for outputManifestItemText in outputManifestItems:
            eoutputManifestItem = json.loads(outputManifestItemText)

            imagesAndLabels = json.loads(S3Helper.readFromS3Uri(eoutputManifestItem["source-ref"]))
            
            i = 0
            for imageAndLabel in imagesAndLabels:
                if(eoutputManifestItem['labels']['item-{}'.format(i)] == "Yes"):
                    if(imageAndLabel["imageUrl"] in finalManifestItems):
                        finalManifestItem = finalManifestItems[imageAndLabel["imageUrl"]]
                    else:
                        finalManifestItem = []
                        finalManifestItems[imageAndLabel["imageUrl"]] = finalManifestItem
                        
                    finalManifestItem.append({"label": imageAndLabel["label"], "confidence": 1})
                    
                i += 1
        
        self.generateOutput(finalManifestItems, labelVerificationJobName)

    def checkJobStatus(self, labelVerificationJobName):
        sageMakerClient = AwsHelper().getClient("sagemaker", self.inputParameters["awsRegion"])
        response = sageMakerClient.describe_labeling_job(LabelingJobName=labelVerificationJobName)
        print("Job: {}, Status: {}".format(response["LabelingJobName"], response["LabelingJobStatus"]))
        jobStatus = response["LabelingJobStatus"]
        while (jobStatus == "InProgress"):
            time.sleep(10)
            response = sageMakerClient.describe_labeling_job(LabelingJobName=labelVerificationJobName)
            print("Job: {}, Status: {}".format(response["LabelingJobName"], response["LabelingJobStatus"]))
            jobStatus = response["LabelingJobStatus"]


    def run(self, labelVerificationJob):
        self.checkJobStatus(labelVerificationJob)
        self.processJobResults(labelVerificationJob)

class JobProcessor:

    def __init__(self):
        self.inputParameters = {}

    def mergeBBAndLabelsOutput(self, hasBBResults, hasLabelResults, hasNoLabelResults):

        lbManifestItems = ""
        bbManifestItems = ""
        noLabelsManifestItems = ""

        if(hasLabelResults):
            lbOutputText = S3Helper.readFromS3(self.inputParameters["outputBucket"], self.inputParameters["labelsOutputFile"])
            lbManifestItems = lbOutputText.splitlines()

        if(hasBBResults):
            bbOutputText = S3Helper.readFromS3(self.inputParameters["outputBucket"], self.inputParameters["boundingBoxOutputFile"])
            bbManifestItems = bbOutputText.splitlines()

        if(hasNoLabelResults):
            noLabelsText = S3Helper.readFromS3(self.inputParameters["outputBucket"], self.inputParameters["noLabelsFile"])
            noLabelsManifestItems = noLabelsText.splitlines()

        bblb = {}

        for lbManifestItemLine in lbManifestItems:
            lbManifestItem = json.loads(lbManifestItemLine)
            bblb[lbManifestItem["source-ref"]] = lbManifestItem
            
        for bbManifestItemLine in bbManifestItems:
            bbManifestItem = json.loads(bbManifestItemLine)
            if (bbManifestItem["source-ref"] in bblb):
                item = bblb[bbManifestItem["source-ref"]]
                for ekey in bbManifestItem:
                    if(not ekey == "source-ref"):
                        item[ekey] = bbManifestItem[ekey]
            else:
                bblb[bbManifestItem["source-ref"]] = bbManifestItem

        for noLabelManifestItemLine in noLabelsManifestItems:
            noLabelManifestItem = json.loads(noLabelManifestItemLine)
            bblb[noLabelManifestItem["source-ref"]] = noLabelManifestItem

        bblbText = ""
        for bblbItem in bblb:
            bblbText += json.dumps(bblb[bblbItem]) + "\n"

        S3Helper.writeToS3(bblbText, self.inputParameters["outputBucket"], self.inputParameters["bblbOutputFile"])

        print("\nOutput\n=====================")
        print("Presigned Url:")
        print(S3Helper.generatePresignedUrl(self.inputParameters["outputBucket"], self.inputParameters["bblbOutputFile"]))
        print("\nS3 Path:")
        print("s3://{}/{}".format(self.inputParameters["outputBucket"], self.inputParameters["bblbOutputFile"]))

    def processJobFile(self):
        
        # print("s3: {}, file: {}".format(self.inputParameters["outputBucket"], self.inputParameters["jobsListFile"]))

        jobs = json.loads(S3Helper.readFromS3(self.inputParameters["outputBucket"], self.inputParameters["jobsListFile"]))

        runid = jobs["runid"]
        boundingBoxOutputPath = "datasets/{}/bounding-box-verification/output".format(runid)
        labelsAndBoundingBoxOutputPath = "datasets/{}/output".format(runid)
        labelOutputPath = "datasets/{}/label-verification/output".format(runid)
        labelsOutputFile = "{}/labels-output.manifest".format(labelOutputPath)
        boundingBoxOutputFile = "{}/bounding-box-output.manifest".format(boundingBoxOutputPath)
        bblbOutputFile = "{}/output.manifest".format(labelsAndBoundingBoxOutputPath)
        self.inputParameters["awsRegion"] = S3Helper.getS3BucketRegion(self.inputParameters["outputBucket"])
        self.inputParameters["labelsOutputFile"] = labelsOutputFile
        self.inputParameters["boundingBoxOutputFile"] = boundingBoxOutputFile
        self.inputParameters["bblbOutputFile"] = bblbOutputFile

        return jobs       

    def validateInput(self, args):
        event = {}
        i = 0
        while(i < len(args)):
            if(args[i] == '--jobs-manifest'):
                event['jobsFile'] = args[i+1]
            i += 1
        return event

    def run(self, args):
        event = self.validateInput(args)
        self.jobsFile = event["jobsFile"]
        print("Jobs manifest file: {}".format(self.jobsFile))

        outputBucket, jobsListFile = S3Helper.parseBucketAndDocumentName(self.jobsFile)
        # print("Output bucket: {}, Jobs file: {}".format(outputBucket, jobsListFile))
        self.inputParameters["outputBucket"] = outputBucket
        self.inputParameters["jobsListFile"] = jobsListFile

        jobs = self.processJobFile()

        hasLabelResults = False
        if(jobs["label-verification-job"]):
            print("Processing label verification jobs...")
            labelJobProcessor = LabelVerificationJobProcessor(self.inputParameters)
            labelJobProcessor.run(jobs["label-verification-job"])
            hasLabelResults = True
            print("Processed label verification jobs...")

        hasBBResults = False
        if(jobs["bounding-box-verification-jobs"]):
            print("Processing bounding box verification jobs...")
            bbJobProcessor = BoundingBoxVerificationJobProcessor(self.inputParameters)
            bbJobProcessor.run(jobs["bounding-box-verification-jobs"])
            hasBBResults = True
            print("Processed bounding box verification jobs...")

        hasNoLabelResults = False
        if(jobs["no-labels-manifest-file"]):
            print("Processing no labels manifest...")
            # print(jobs["no-labels-manifest-file"])
            nlbBucket, self.inputParameters["noLabelsFile"] = S3Helper.parseBucketAndDocumentName(jobs["no-labels-manifest-file"])
            hasNoLabelResults = True
            print("Processed no labels manifest...")

        self.mergeBBAndLabelsOutput(hasBBResults, hasLabelResults, hasNoLabelResults)

# try:
cliMode = True

if cliMode:
    args = sys.argv
else:
    args = runCommand.split(' ')

jobProcessor = JobProcessor()
jobProcessor.run(args)

# except Exception as e:
#     print("Something went wrong:\n====================================================\n{}".format(e)