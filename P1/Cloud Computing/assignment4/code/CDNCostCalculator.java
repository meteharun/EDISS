import java.io.IOException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class CDNCostCalculator {

    public static class LogMapper extends Mapper<LongWritable, Text, Text, LongWritable> {
        private final static Text requestsKey = new Text("requests");
        private final static Text dataKey = new Text("dataTransferred");

        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String[] fields = value.toString().split(" ");
            if (fields.length > 0) {
                // Count each line as a request
                context.write(requestsKey, new LongWritable(1));
                
                // Get the last field as the size in bytes
                try {
                    long responseSize = Long.parseLong(fields[fields.length - 1]);
                    context.write(dataKey, new LongWritable(responseSize));
                } catch (NumberFormatException e) {
                    // Ignore if unable to parse; no action needed since no exceptions should be present
                }
            }
        }
    }

    public static class LogReducer extends Reducer<Text, LongWritable, Text, DoubleWritable> {
        private final static double REQUEST_COST = 0.001;
        private final static double DATA_COST_PER_GB = 0.08;
        private final static double BYTES_IN_GB = 1024 * 1024 * 1024;

        private long totalRequests = 0;
        private long totalDataBytes = 0;

        public void reduce(Text key, Iterable<LongWritable> values, Context context) throws IOException, InterruptedException {
            long sum = 0;
            for (LongWritable val : values) {
                sum += val.get();
            }

            if (key.toString().equals("requests")) {
                totalRequests = sum;
            } else if (key.toString().equals("dataTransferred")) {
                totalDataBytes = sum;
            }
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            // Convert total data from bytes to GB
            double totalDataGB = totalDataBytes / BYTES_IN_GB;

            // Calculate costs
            double requestCost = totalRequests * REQUEST_COST;
            double dataCost = totalDataGB * DATA_COST_PER_GB;
            double totalCost = requestCost + dataCost;

            // Write the results
            context.write(new Text("num_of_requests"), new DoubleWritable(totalRequests));
            context.write(new Text("transferred_data (GB)"), new DoubleWritable(totalDataGB));
            context.write(new Text("cost (EUR)"), new DoubleWritable(totalCost));
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "CDN cost calculator");
        job.setJarByClass(CDNCostCalculator.class);
        job.setMapperClass(LogMapper.class);
        job.setReducerClass(LogReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(LongWritable.class);
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
