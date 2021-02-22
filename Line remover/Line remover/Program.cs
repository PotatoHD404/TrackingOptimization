using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;

namespace Line_remover
{
    class Program
    {
        static void Main(string[] args)
        {
            var path = @"winSize.csv";
            var data = File.ReadAllLines(path).ToList();
            Console.WriteLine(data.Count);
            Dictionary<string, int> dict = new Dictionary<string, int>();
            for (int i = 0; i < data.Count; i++)
            {
                if (!data[i].Contains(",,,,,,,,,,,,,,,,,,,,"))
                {
                    var keys1 = data[i].Replace(",,,,,,,,,,,,,,,,,,,,", "").Split(",");
                    for (int j = 0; j < keys1.Length; j++)
                    {
                        if (!dict.ContainsKey(keys1[j]))
                        {
                            dict.Add(keys1[j], -1);
                        }
                        else
                        {
                            dict[keys1[j]]--;
                        }
                    }
                    continue;
                }
                var keys = data[i].Replace(",,,,,,,,,,,,,,,,,,,,", "").Split(",");
                    for (int j = 0; j < keys.Length; j++)
                    {
                        if (!dict.ContainsKey(keys[j]))
                        {
                            dict.Add(keys[j], 1);
                        }
                        else
                        {
                            dict[keys[j]]++;
                        }
                    }
                

                data.RemoveAt(i);
                    i--;
                
            }

            foreach (var t in dict)
            {
                if(t.Value == -1)
                    Console.WriteLine(t.Key + " " + t.Value);
            }
            Console.WriteLine(data.Count);
            File.WriteAllLines(path.Replace(".csv", "_edited.csv"), data);
            Console.WriteLine("Done");
        }
    }
}
