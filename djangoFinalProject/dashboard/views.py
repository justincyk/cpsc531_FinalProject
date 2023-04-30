from django.shortcuts import render


def mushrooms_report_page(request):
   return render(request, 'mushrooms_report.html', {})
