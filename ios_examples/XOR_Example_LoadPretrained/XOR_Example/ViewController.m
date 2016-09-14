//
//  ViewController.m
//  XOR_Example
//
//  Created by Kurt Jacobs on 2015/06/12.
//  Copyright (c) 2015 RandomDudes. All rights reserved.
//

#import "ViewController.h"

#define KBYTES_CLEAN_UP 10000 //10 Megabytes Max Storage Otherwise Force Cleanup (For This Example We Will Probably Never Reach It -- But Good Practice).
#define LUAT_STACK_INDEX_FLOAT_TENSORS 4 //Index of Garbage Collection Stack Value

#define CLAMP(value, min, max) MAX(MIN(value, max), min)

typedef struct {
    uint8_t red;
    uint8_t green;
    uint8_t blue;
    uint8_t alpha;
} rgba;

@interface ViewController ()

@end

@implementation ViewController

- (void)viewDidLoad
{
    [super viewDidLoad];
  
    UITapGestureRecognizer *tap = [[UITapGestureRecognizer alloc] initWithTarget:self action:@selector(disableKeyboard)];
    [self.view addGestureRecognizer:tap];
  
    self.t = [Torch new];
    [self.t initialize];
    [self.t runMain:@"main" inFolder:@"xor_lua"];
    [self.t loadFileWithName:@"model.t7" inResourceFolder:@"xor_lua" andLoadMethodName:@"loadNeuralNetwork"];
}

- (void)disableKeyboard
{
  [self.valueTwoTextField resignFirstResponder];
  [self.valueOneTextfield resignFirstResponder];
}

- (IBAction)classifyAction:(id)sender
{
  if ([self isValidFloat:self.valueOneTextfield.text] && [self isValidFloat:self.valueTwoTextField.text])
  {
    float v1 = [self.valueOneTextfield.text floatValue];
    float v2 = [self.valueTwoTextField.text floatValue];
    [self perfClassificationOnValuesv1:v1 v2:v2];
  }
  else
  {
    self.answerLabel.text = @"Please Enter Valid Floats!!!";
  }
}

- (BOOL)isValidFloat:(NSString*)string
{
  NSScanner *scanner = [NSScanner scannerWithString:string];
  [scanner scanFloat:NULL];
  return [scanner isAtEnd];
}

- (void)perfClassificationOnValuesv1:(float)v1 v2:(float)v2
{
  XORClassifyObject *classificationObj = [XORClassifyObject new];
  classificationObj.x = v1;
  classificationObj.y = v2;
  float value = [self classifyExample:classificationObj inState:[self.t getLuaState]];
  self.answerLabel.text = [NSString stringWithFormat:@"Classification Value: %f",value];
}

- (CGFloat)classifyExample:(XORClassifyObject *)obj inState:(lua_State *)L
{
    NSInteger garbage_size_kbytes = lua_gc(L, LUA_GCCOUNT, LUAT_STACK_INDEX_FLOAT_TENSORS);
    printf("memory used by lua: %ldK\n", (long)garbage_size_kbytes);
    if (garbage_size_kbytes >= KBYTES_CLEAN_UP)
    {
        NSLog(@"LUA -> Cleaning Up Garbage");
        lua_gc(L, LUA_GCCOLLECT, LUAT_STACK_INDEX_FLOAT_TENSORS);
    }
    
    NSString *filePath = [[NSBundle mainBundle] pathForResource:@"lena" ofType:@"jpg"];
    NSData *imageData = [NSData dataWithContentsOfFile:filePath];
    UIImage *image = [UIImage imageWithData:imageData];
    
    int imageWidth = (int)CGImageGetWidth(image.CGImage);
    int imageHeight = (int)CGImageGetHeight(image.CGImage);
    
    THFloatStorage *input_storage = THFloatStorage_newWithSize4(1, 3, imageHeight, imageWidth);
    THFloatTensor *input = THFloatTensor_newWithStorage4d(input_storage, 0, 1, 3 * imageHeight * imageWidth, 3, imageHeight * imageWidth, imageHeight, imageWidth, imageWidth, 1);
    
    int resultWidth = imageWidth + 3;
    int resultHeight = imageHeight + 3;
    
    THFloatStorage *output_storage = THFloatStorage_newWithSize4(1, 3, resultHeight, resultWidth);
    THFloatStorage_fill(output_storage, 0.0f);
    THFloatTensor *output = THFloatTensor_newWithStorage4d(output_storage, 0, 1, 3 * resultHeight * resultWidth, 3, resultHeight * resultWidth, resultHeight, resultWidth, resultWidth, 1);
    
    // First get the image into your data buffer
    CGImageRef imageRef = [image CGImage];
    NSUInteger width = CGImageGetWidth(imageRef);
    NSUInteger height = CGImageGetHeight(imageRef);
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    unsigned char *rawData = (unsigned char*) calloc(height * width * 4, sizeof(unsigned char));
    NSUInteger bytesPerPixel = 4;
    NSUInteger bytesPerRow = bytesPerPixel * width;
    NSUInteger bitsPerComponent = 8;
    CGContextRef context = CGBitmapContextCreate(rawData, width, height,
                                                 bitsPerComponent, bytesPerRow, colorSpace,
                                                 kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
    
    CGContextDrawImage(context, CGRectMake(0, 0, width, height), imageRef);
    CGContextRelease(context);
    
    for (int y = 0; y < imageHeight; y++) {
        for (int x = 0; x < imageWidth; x++) {
            int offset = y * imageWidth * 4 + x * 4;
            
            float r = ((float)rawData[offset] / 255.0);
            float g = ((float)rawData[offset + 1] / 255.0);
            float b = ((float)rawData[offset + 2] / 255.0);
            
//            NSLog(@"rgb = (%@, %@, %@)", @(r), @(g), @(b));
            
            THTensor_fastSet4d(input, 0, 0, y, x, r);
            THTensor_fastSet4d(input, 0, 1, y, x, g);
            THTensor_fastSet4d(input, 0, 2, y, x, b);
        }
    }
    
    lua_getglobal(L,"classifyExample");
    luaT_pushudata(L, input, "torch.FloatTensor");
    luaT_pushudata(L, output, "torch.FloatTensor");
    
    NSString *argFilePath = [NSString stringWithFormat:@"%@/", NSTemporaryDirectory()];
    const char *filePathCString = [argFilePath UTF8String];
    lua_pushstring(L, filePathCString);
    
    // NN forward operation
    NSDate *start = [NSDate date];
    //p_call -- args, results
    int res = lua_pcall(L, 3, 1, 0);
    NSTimeInterval timeInterval = fabs([start timeIntervalSinceNow]);
    NSLog(@"Forward took %.2f sec", timeInterval);
    
    if (res != 0)
    {
        NSLog(@"error running function `f': %s",lua_tostring(L, -1));
        return -1;
    }
    
    unsigned char *resultRawData = (unsigned char*) calloc(resultWidth * resultHeight * 4, sizeof(unsigned char));
    
    for (int y = 0; y < resultHeight; y++) {
        for (int x = 0; x < resultWidth; x++) {
            int offset = y * resultWidth * 4 + x * 4;
            
            float r = THTensor_fastGet4d(output, 0, 2, y, x) + 103.939;
            float g = THTensor_fastGet4d(output, 0, 1, y, x) + 116.779;
            float b = THTensor_fastGet4d(output, 0, 0, y, x) + 123.68;
            
            r = CLAMP(r, 0, 255);
            g = CLAMP(g, 0, 255);
            b = CLAMP(b, 0, 255);
            
            resultRawData[offset] = (uint8_t)r;
            resultRawData[offset + 1] = (uint8_t)g;
            resultRawData[offset + 2] = (uint8_t)b;
            resultRawData[offset + 3] = (uint8_t)255;
        }
    }
    
    NSData *outputData = [NSData dataWithContentsOfFile:argFilePath];
    
    bytesPerRow = resultWidth * 4;
    CGContextRef newContext = CGBitmapContextCreate(resultRawData, resultWidth, resultHeight,
                                                 bitsPerComponent, bytesPerRow, colorSpace,
                                                 kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
    
    CGImageRef resultImageRef = CGBitmapContextCreateImage(newContext);
    UIImage *result = [UIImage imageWithCGImage:resultImageRef];
    CGContextRelease(newContext);
    //set breakpoint here to see image
    CGColorSpaceRelease(colorSpace);
    
    free(rawData);
    free(resultRawData);
  
  if (!lua_isnumber(L, -1))
  {
    NSLog(@"function `f' must return a number");
  }
//    int nresult = lua_gettop(L);
    
  CGFloat returnValue = lua_tonumber(L, -1);
  lua_pop(L, 1);  /* pop returned value */
  return returnValue;
}

- (void)didReceiveMemoryWarning
{
  [super didReceiveMemoryWarning];
}

@end
