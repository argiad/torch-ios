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
    [self.t loadFileWithName:@"xor_model.net" inResourceFolder:@"xor_lua" andLoadMethodName:@"loadNeuralNetwork"];
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
    
  THFloatStorage *classification_storage = THFloatStorage_newWithSize4(1, 3, imageWidth, imageHeight);
  THFloatTensor *classification = THFloatTensor_newWithStorage4d(classification_storage, 0, 1, 1, 3, 1, imageWidth, 1, imageHeight, 1);
    
    
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
            
            THTensor_fastSet4d(classification, 0, 0, x, y, r);
            THTensor_fastSet4d(classification, 0, 1, x, y, g);
            THTensor_fastSet4d(classification, 0, 2, x, y, b);
        }
    }
    
//  THTensor_fastSet1d(classification, 0, obj.x);
//  THTensor_fastSet1d(classification, 1, obj.y);
  lua_getglobal(L,"classifyExample");
  luaT_pushudata(L, classification, "torch.FloatTensor");
  NSDate *start = [NSDate date];
  //p_call -- args, results
    
  int res = lua_pcall(L, 1, 1, 0);
  NSTimeInterval timeInterval = fabs([start timeIntervalSinceNow]);
  NSLog(@"Forward took %.2f sec", timeInterval);
  if (res != 0)
  {
    NSLog(@"error running function `f': %s",lua_tostring(L, -1));
      return -1;
  }
    
    
    CGContextRef newContext = CGBitmapContextCreate(rawData, width, height,
                                                 bitsPerComponent, bytesPerRow, colorSpace,
                                                 kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
    
    CGImageRef resultImageRef = CGBitmapContextCreateImage(context);
//    CGContextDrawImage(newContext, CGRectMake(0, 0, width, height), imageRef);
//    UIImage *result = UIGraphicsGetImageFromCurrentImageContext();
    UIImage *result = [UIImage imageWithCGImage:resultImageRef];
    CGContextRelease(newContext);
    CGColorSpaceRelease(colorSpace);
    
  
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
